import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from utilsLiteLLM import CachedLLM
from utilsOutput import construct_messages

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGPROB_MODELS = [
    "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano", "gpt_5.1",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct", "Apertus_8B_Instruct",
]

SYSTEM_PROMPT = "You are a participant in a behavioral experiment. Follow the instruction exactly."


def get_response_tokens(response, task_value: str) -> tuple[list[str], list[float], str]:
    """Return (tokens, per_token_logprobs, notes) for the generated response,
    handling both the litellm-normalised `lp.content` shape and the
    `lp.tokens`/`lp.top_logprobs` shape some HF backends use instead
    (mirrors the two branches in utilsOutput.get_probs_task).
    """
    lp = getattr(response.choices[0], "logprobs", None)
    if lp is None:
        return [], [], "no logprobs returned for this response"

    if lp.content is not None:
        tokens = [str(t.token) for t in lp.content]
        logprobs = [t.logprob for t in lp.content]
    else:
        tokens = [str(t) for t in lp.tokens]
        logprobs = list(lp.token_logprobs) if getattr(lp, "token_logprobs", None) else [None] * len(tokens)

    text = response.choices[0].message.content or ""
    notes = "" if text.strip() == task_value else f"generated text {text!r} does not exactly match requested value {task_value!r}"
    return tokens, logprobs, notes


def verify_model(model_id: str, model_info: dict, values: list[int], db_path: str) -> list[dict]:
    model = model_info["provider"] + "/" + model_info["model"]
    call_kwargs = {k: v for k, v in model_info["kwargs"].items() if k not in ("n",)}
    call_kwargs["n"] = 1  # one token sequence to inspect per value is enough here

    llm = CachedLLM(model=model, db_path=db_path, memory_cache=True)

    rows = []
    for v in values:
        task_value = str(v)
        messages = construct_messages(
            SYSTEM_PROMPT,
            f"Respond with exactly this number and nothing else: {task_value}",
        )
        row = {"model": model_id, "value": v}
        try:
            response = llm.complete(messages=messages, sample_idx=v, **call_kwargs)
            tokens, logprobs, notes = get_response_tokens(response, task_value)
            row["tokens"] = json.dumps(tokens)
            row["token_logprobs"] = json.dumps(logprobs)
            row["num_tokens"] = len(tokens)
            row["single_token"] = len(tokens) == 1
            row["notes"] = notes
        except Exception as e:
            row["notes"] = f"call failed: {e}"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Verify, per active logprob model, whether each integer 0-30 (which "
                    "already covers every default anchor: 0, 11, 18, 23, 26, 30) encodes as "
                    "a single token at the answer position. Multi-token values need the "
                    "product-over-token-sequence probability instead of a plain first-token "
                    "logprob read."
    )
    parser.add_argument("--models_path", type=str, default=os.path.join(PROJECT_DIR, "data", "models.json"))
    parser.add_argument("--db_path", type=str, default=os.path.join(PROJECT_DIR, "llm_cache.db"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_DIR, "output", "tokenization_verification.csv"))
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Subset of LOGPROB_MODELS to check (default: all of them)")
    args = parser.parse_args()

    with open(args.models_path, encoding="utf-8") as f:
        model_list = json.load(f)

    values = list(range(0, 31))
    model_ids = args.models or LOGPROB_MODELS

    all_rows = []
    for model_id in model_ids:
        print(f"Verifying tokenization for {model_id}...")
        rows = verify_model(model_id, model_list[model_id], values, args.db_path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSaved tokenization verification ({len(df)} rows) to: {args.output}")
    if "single_token" in df.columns:
        summary = df.groupby("model")["single_token"].agg(["sum", "count"])
        summary["pct_single_token"] = (summary["sum"] / summary["count"] * 100).round(1)
        print("\nPer-model single-token rate (of 31 values, 0-30):")
        print(summary.rename(columns={"sum": "single_token_count", "count": "n_values"}))
        multi = df[df["single_token"] == False]  # noqa: E712
        if not multi.empty:
            print("\nModels/values needing product-over-sequence probability:")
            print(multi[["model", "value", "num_tokens", "tokens"]].to_string(index=False))


if __name__ == "__main__":
    main()
