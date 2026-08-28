"""Rerun just V0_true (the actual main-run default-v0.txt wording, with the
'Your extraction this round: {D}.' lead-in) at both anchors (18, 26) for all
6 models. The prompt-robustness check's "V0" was a shortened reword, not
this sentence - this fills in the true reference point without touching
V1-V4 or any other anchor.
"""
import json
import os

from prompt_robustness_check import (
    MODELS, ANCHORS, VARIANT_TEMPLATES, CHOICES, CHOICES_PADDED,
    OUTPUT_DIR, PROJECT_DIR, TASK, VERSION, SYSTEM_TYPE,
    load_model_config, build_messages, run_cell,
)
from utilsLiteLLM import CachedLLM

VARIANT = "V0_true"


def main():
    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-instructions-v{VERSION}.txt"), encoding="utf-8") as f:
        instructions = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-table.txt"), encoding="utf-8") as f:
        cpr_table = f.read()

    rows = []
    for model_id in MODELS:
        model, kwargs = load_model_config(model_id)
        llm = CachedLLM(model=model, db_path=os.path.join(PROJECT_DIR, "llm_cache.db"), memory_cache=True)

        for d in ANCHORS:
            default_line = VARIANT_TEMPLATES[VARIANT].format(D=d)
            messages = build_messages(system, instructions, cpr_table, default_line)

            print(f"Running {model_id} default={d} {VARIANT}...")
            result = run_cell(llm, kwargs, messages, padded=False)

            fname = f"{model_id}-{TASK}-default={d}-{VARIANT}-v{VERSION}-n1-{SYSTEM_TYPE}.json"
            path = os.path.join(OUTPUT_DIR, fname)
            out = {
                "model": model_id, "default": d, "variant": VARIANT,
                "default_line": default_line, "padded_format": False,
            }
            if "error" in result:
                out["error"] = result["error"]
                print(f"  FAILED: {result['error']}")
                rows.append([model_id, d, VARIANT, "", "", "", "", result["error"]])
            else:
                out["cpr"] = [{"sample-0": result["probs"]}]
                out["cpr_raw_logprobs"] = [{"sample-0": {
                    "content": result["content"],
                    "positions": result["positions_raw"],
                }}]
                out["coverage"] = result["coverage"]
                out["argmax"] = result["argmax"]
                out["argmax_prob"] = result["argmax_prob"]
                spike = result["probs"].get(str(d))
                print(f"  spike@{d}={spike!r}  argmax={result['argmax']} ({result['argmax_prob']:.3f})  "
                     f"coverage={result['coverage']:.3f}")
                rows.append([model_id, d, VARIANT, spike, result["argmax"], result["argmax_prob"], result["coverage"], ""])
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    # append to the existing summary.csv
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    import csv
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    print(f"\nAppended {len(rows)} rows to {summary_path}")


if __name__ == "__main__":
    main()
