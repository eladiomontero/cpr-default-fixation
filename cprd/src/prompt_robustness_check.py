"""Prompt-robustness check on the two load-bearing anchors (default=18 Nash,
default=26 indefensible) of the CPR default-fixation result: are the
near-universal following of 18 and the model split at 26 stable across
trivial rewordings of the default-instruction line, or artefacts of one
specific wording?

6 models x 2 anchors x 5 variants = 60 logprob calls. No sampling fallback -
a cell with no logprobs is reported FAILED, never substituted.

Raw per-position (0 and 1) top-k logprobs are persisted in every saved JSON
alongside the reconstructed p_0..p_30, so this is reproducible from the
output/prompt_robustness/ JSONs without ever re-hitting the API.
"""
import json
import math
import os

from utilsLiteLLM import CachedLLM
from utilsOutput import _reconstruct_number_probs, construct_messages

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "cpr"
VERSION = "0"
SYSTEM_TYPE = "blinded"

MODELS = [
    "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
]
ANCHORS = [18, 26]

VARIANT_TEMPLATES = {
    # True main-run wording (prompts/cpr/default-v0.txt, %i -> {D}). Every
    # variant below KEEPS this exact lead-in and varies only the sentence
    # after it, so a delta vs V0_true isolates the suggestion wording, not
    # the presence/absence of the lead-in (that was a confound in the first
    # pass - proven by Llama's spike@26 moving 0.007 -> 1.00 on the lead-in
    # alone, not the suggestion phrasing).
    "V0_true": "Your extraction this round: {D}. This value is pre-selected for you; you can change it if you want.",
    "V1": "Your extraction this round: {D}. The value {D} has been pre-filled; you may keep it or choose another.",
    "V2": "Your extraction this round: {D}. A value of {D} is shown by default; select whatever number you prefer.",
    "V3": "Your extraction this round: {D}. The system suggests {D}; choose your preferred number.",
    "V4": ("Your extraction this round: {D}. This value is pre-selected for you; you can change it if you want.\n\n"
          "Answer format: respond with EXACTLY TWO DIGITS (00 to 30), e.g. \"05\" for five, "
          "\"20\" for twenty. No other characters."),
}
VARIANTS = list(VARIANT_TEMPLATES.keys())

CHOICES = [str(i) for i in range(0, 31)]
CHOICES_PADDED = [f"{i:02d}" for i in range(0, 31)]

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "prompt_robustness")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_positions(choice) -> list[tuple[str, float, list[tuple[str, float]]]]:
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        raise RuntimeError("No logprobs returned")
    if lp.content is None:  # huggingface-style parallel arrays
        return [
            (str(tok), None, [(str(t), v) for t, v in lp.top_logprobs[i].items()])
            for i, tok in enumerate(lp.tokens)
        ]
    return [
        (str(t.token), t.logprob, [(str(a.token), a.logprob) for a in t.top_logprobs])
        for t in lp.content
    ]


def positions_to_json(positions: list) -> list[dict]:
    out = []
    for tok, lp, alts in positions[:2]:
        out.append({
            "token": tok,
            "logprob": lp,
            "top_logprobs": [{"token": a_tok, "logprob": a_lp} for a_tok, a_lp in alts],
        })
    return out


def load_model_config(model_id: str) -> tuple[str, dict]:
    with open(os.path.join(PROJECT_DIR, "data", "models.json"), encoding="utf-8") as f:
        info = json.load(f)[model_id]
    return info["provider"] + "/" + info["model"], info["kwargs"]


def build_messages(system: str, instructions: str, cpr_table: str, default_line: str) -> list[dict]:
    parts = [instructions, cpr_table, default_line]
    return construct_messages(system, "\n\n".join(p.strip() for p in parts))


def run_cell(llm: CachedLLM, kwargs: dict, messages: list[dict], padded: bool) -> dict:
    """Returns a result dict with either 'error' or the reconstructed metrics."""
    try:
        response = llm.complete(messages=messages, sample_idx=0, **kwargs)
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    choice = response.choices[0]
    content_text = choice.message.content
    choices = CHOICES_PADDED if padded else CHOICES

    if content_text not in choices:
        return {"error": f"invalid answer {content_text!r} (padded={padded})"}

    try:
        positions = extract_positions(choice)
    except RuntimeError as e:
        return {"error": str(e)}

    probs = _reconstruct_number_probs(content_text, positions, choices)
    if padded:
        probs = {str(i): probs[f"{i:02d}"] for i in range(31)}

    coverage = sum(v for v in probs.values() if v is not None)
    valid = {k: v for k, v in probs.items() if v is not None}
    argmax_key, argmax_val = max(valid.items(), key=lambda kv: kv[1]) if valid else (None, None)

    return {
        "content": content_text,
        "probs": probs,
        "coverage": coverage,
        "argmax": argmax_key,
        "argmax_prob": argmax_val,
        "positions_raw": positions_to_json(positions),
    }


def main():
    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-instructions-v{VERSION}.txt"), encoding="utf-8") as f:
        instructions = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-table.txt"), encoding="utf-8") as f:
        cpr_table = f.read()

    results = {}  # (model, anchor, variant) -> result dict

    for model_id in MODELS:
        model, kwargs = load_model_config(model_id)
        llm = CachedLLM(model=model, db_path=os.path.join(PROJECT_DIR, "llm_cache.db"), memory_cache=True)

        for d in ANCHORS:
            for variant in VARIANTS:
                padded = (variant == "V4")
                default_line = VARIANT_TEMPLATES[variant].format(D=d)
                messages = build_messages(system, instructions, cpr_table, default_line)

                print(f"Running {model_id} default={d} {variant}...")
                result = run_cell(llm, kwargs, messages, padded)
                results[(model_id, d, variant)] = result

                fname = f"{model_id}-{TASK}-default={d}-{variant}-v{VERSION}-n1-{SYSTEM_TYPE}.json"
                path = os.path.join(OUTPUT_DIR, fname)
                out = {
                    "model": model_id, "default": d, "variant": variant,
                    "default_line": default_line, "padded_format": padded,
                }
                if "error" in result:
                    out["error"] = result["error"]
                    print(f"  FAILED: {result['error']}")
                else:
                    out["cpr"] = [{"sample-0": result["probs"]}]
                    out["cpr_raw_logprobs"] = [{"sample-0": {
                        "content": result["content"],
                        "positions": result["positions_raw"],
                    }}]
                    out["coverage"] = result["coverage"]
                    out["argmax"] = result["argmax"]
                    out["argmax_prob"] = result["argmax_prob"]
                    print(f"  spike@{d}={result['probs'].get(str(d))!r}  "
                         f"argmax={result['argmax']} ({result['argmax_prob']:.3f})  "
                         f"coverage={result['coverage']:.3f}")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

    # --------------------------------------------------------------------
    # Persist a flat summary CSV for downstream reporting
    # --------------------------------------------------------------------
    import csv
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "default", "variant", "spike", "argmax", "argmax_prob", "coverage", "error"])
        for (model_id, d, variant), r in results.items():
            if "error" in r:
                w.writerow([model_id, d, variant, "", "", "", "", r["error"]])
            else:
                spike = r["probs"].get(str(d))
                w.writerow([model_id, d, variant, spike, r["argmax"], r["argmax_prob"], r["coverage"], ""])
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
