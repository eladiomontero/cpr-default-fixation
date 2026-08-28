import argparse
import json
import os

import pandas as pd

from utilsLiteLLM import CachedLLM
from utilsOutput import construct_messages

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only re-verify models where logprobs=true - top_logprobs has no meaning
# for the sampling-fallback (logprobs=false) models.
LOGPROB_MODELS = [
    "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano", "gpt_5.1",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct", "Apertus_8B_Instruct",
]

FALLBACK_DEPTH = {
    "Llama_3.3_70B_Instruct": 1,
    "Qwen3_235B_A22B_Instruct": 1,
    "Apertus_8B_Instruct": 1,
}


def probe_depth(llm: CachedLLM, messages: list[dict], call_kwargs: dict, requested: int) -> tuple[int, str]:
    """Try `requested` top_logprobs; on rejection, retry with the model's
    documented fallback depth (or halve down to 1 as a last resort if none
    is known). Returns (actual_returned_depth, notes).
    """
    depths_to_try = [requested]
    fallback = FALLBACK_DEPTH.get(call_kwargs.pop("_model_id", None))

    def attempt(depth):
        kwargs = {k: v for k, v in call_kwargs.items() if not k.startswith("_")}
        kwargs["top_logprobs"] = depth
        response = llm.complete(messages=messages, sample_idx=depth, **kwargs)
        lp = response.choices[0].logprobs
        if lp is None:
            raise RuntimeError("no logprobs returned at all")
        if lp.content is not None:
            returned = len(lp.content[0].top_logprobs)
        else:
            returned = len(lp.top_logprobs[0])
        return returned

    try:
        returned = attempt(requested)
        notes = "" if returned == requested else f"requested {requested}, backend silently returned {returned}"
        return returned, notes
    except Exception as e:
        if fallback is None:
            return 0, f"requested {requested} rejected ({e}); no documented fallback depth to retry"
        try:
            returned = attempt(fallback)
            return returned, f"requested {requested} rejected ({e}); fell back to documented depth {fallback}, backend returned {returned}"
        except Exception as e2:
            return 0, f"requested {requested} rejected ({e}); fallback {fallback} also rejected ({e2})"


def main():
    parser = argparse.ArgumentParser(
        description="Probe each active logprob model's actual accepted top_logprobs depth "
                    "against the uniformly requested value (20), falling back to the "
                    "provider's documented cap on rejection. Does not modify models.json - "
                    "report the results and update the relevant _note/top_logprobs by hand."
    )
    parser.add_argument("--requested", type=int, default=5,
                        help="5 is the confirmed hard ceiling for both OpenAI and HF's router - 20 was tried and rejected everywhere")
    parser.add_argument("--models_path", type=str, default=os.path.join(PROJECT_DIR, "data", "models.json"))
    parser.add_argument("--db_path", type=str, default=os.path.join(PROJECT_DIR, "llm_cache.db"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_DIR, "output", "top_logprobs_verification.csv"))
    args = parser.parse_args()

    with open(args.models_path, encoding="utf-8") as f:
        model_list = json.load(f)

    system_prompt = "You are a participant in a behavioral experiment. Respond with only a single integer between 0 and 30."
    messages = construct_messages(system_prompt, "Pick exactly one number between 0 and 30.")

    rows = []
    for model_id in LOGPROB_MODELS:
        info = model_list[model_id]
        model = info["provider"] + "/" + info["model"]
        llm = CachedLLM(model=model, db_path=args.db_path, memory_cache=True)
        call_kwargs = {k: v for k, v in info["kwargs"].items() if k not in ("top_logprobs",)}
        call_kwargs["_model_id"] = model_id

        try:
            returned, notes = probe_depth(llm, messages, call_kwargs, args.requested)
        except Exception as e:
            returned, notes = 0, f"probe failed: {e}"

        rows.append({
            "model": model_id,
            "requested_top_logprobs": args.requested,
            "actual_returned_depth": returned,
            "notes": notes,
        })
        print(f"{model_id}: requested={args.requested} -> returned={returned}  {notes}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved verification report to: {args.output}")


if __name__ == "__main__":
    main()
