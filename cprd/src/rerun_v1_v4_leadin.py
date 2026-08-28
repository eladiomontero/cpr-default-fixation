"""Rerun V1-V4 with the main-run lead-in restored (see VARIANT_TEMPLATES in
prompt_robustness_check.py). V0_true is NOT rerun here - it already exists
as cache hits against the frozen main-run data from rerun_v0_true.py.

Sanity gate: prints the full built prompt for one cell (gpt_5.4, anchor=26,
V1) before running anything, so the lead-in's presence can be visually
confirmed rather than assumed.
"""
import csv
import json
import os

from prompt_robustness_check import (
    MODELS, ANCHORS, VARIANT_TEMPLATES, OUTPUT_DIR, PROJECT_DIR,
    TASK, VERSION, SYSTEM_TYPE, load_model_config, build_messages, run_cell,
)
from utilsLiteLLM import CachedLLM

VARIANTS_TO_RUN = ["V1", "V2", "V3", "V4"]


def main():
    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-instructions-v{VERSION}.txt"), encoding="utf-8") as f:
        instructions = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-table.txt"), encoding="utf-8") as f:
        cpr_table = f.read()

    # --- SANITY GATE ---
    check_line = VARIANT_TEMPLATES["V1"].format(D=26)
    check_messages = build_messages(system, instructions, cpr_table, check_line)
    check_user_prompt = check_messages[1]["content"]
    print("=== SANITY GATE: gpt_5.4, anchor=26, V1, full user-prompt string ===")
    print(check_user_prompt)
    print("=== END PROMPT ===")
    if "Your extraction this round:" not in check_user_prompt:
        print("STOP: lead-in NOT present. Aborting before any calls.")
        return
    print("Lead-in confirmed present. Proceeding with 48 calls.\n")

    rows = []
    for model_id in MODELS:
        model, kwargs = load_model_config(model_id)
        llm = CachedLLM(model=model, db_path=os.path.join(PROJECT_DIR, "llm_cache.db"), memory_cache=True)

        for d in ANCHORS:
            for variant in VARIANTS_TO_RUN:
                padded = (variant == "V4")
                default_line = VARIANT_TEMPLATES[variant].format(D=d)
                messages = build_messages(system, instructions, cpr_table, default_line)

                print(f"Running {model_id} default={d} {variant}...")
                result = run_cell(llm, kwargs, messages, padded)

                fname = f"{model_id}-{TASK}-default={d}-{variant}-v{VERSION}-n1-{SYSTEM_TYPE}.json"
                path = os.path.join(OUTPUT_DIR, fname)
                out = {
                    "model": model_id, "default": d, "variant": variant,
                    "default_line": default_line, "padded_format": padded,
                }
                if "error" in result:
                    out["error"] = result["error"]
                    print(f"  FAILED: {result['error']}")
                    rows.append([model_id, d, variant, "", "", "", "", result["error"]])
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
                    rows.append([model_id, d, variant, spike, result["argmax"], result["argmax_prob"], result["coverage"], ""])
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    print(f"\nAppended {len(rows)} rows to {summary_path}")


if __name__ == "__main__":
    main()
