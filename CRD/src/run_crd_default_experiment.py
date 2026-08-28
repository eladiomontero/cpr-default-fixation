"""Default-framing generalization test on a Collective Risk Dilemma (CRD),
mirroring the CPR default-fixation study so the wording/framing effect can
be checked across game structure (contribute-to-threshold vs
extract-from-pool). Reuses the CPR pipeline's logprob/reconstruction/cache
helpers (utilsLiteLLM, utilsOutput, data/models.json all live in cprd/) -
only the game itself (prompt, action space, payoff rule) is new, under
CRD/.

GAME (one-shot core only; uncertainty/info-cost/proposal variants from the
full CRD literature are deliberately dropped):
  - 4-player group, 50 ECU private endowment each.
  - Contribution to a shared Public Account, EXACTLY one of {0, 2, 4}.
  - Certainty target = 80 ECU group total. >=80: everyone keeps 100% of
    their Private Account. <80: everyone keeps only 10% (loses 90%).
  - Single one-shot decision, same blinded system prompt as CPR.

BENCHMARK READING of the anchor values (documented, not enforced):
  0 = free-ride (selfish; contributes nothing, risks the group missing the
      80 target and everyone losing 90% of their endowment)
  4 = full cooperation (safest single contribution toward the target)
  2 = middle ground

No sampling fallback anywhere - a cell with no logprobs is FAILED, not
substituted. Raw position-0 top-k logprobs are persisted in every saved
JSON.
"""
import csv
import json
import math
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CRD/
CPRD_SRC = os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "src")
CPRD_DATA = os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "data", "models.json")
sys.path.insert(0, CPRD_SRC)

from utilsLiteLLM import CachedLLM  # noqa: E402
from utilsOutput import _reconstruct_number_probs, construct_messages  # noqa: E402

TASK = "crd"
VERSION = "0"
SYSTEM_TYPE = "blinded"

MODELS = [
    "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
]
ANCHORS = [0, 2, 4]
CHOICES = ["0", "2", "4"]

VARIANT_TEMPLATES = {
    # Byte-identical to the CPR study's V0_true/V1/V2/V3, verb
    # extract -> contribute, lead-in kept in every variant so wording
    # differences are never confounded with lead-in presence/absence.
    "V0_true": "Your contribution this round: {D}. This value is pre-selected for you; you can change it if you want.",
    "V1": "Your contribution this round: {D}. The value {D} has been pre-filled; you may keep it or choose another.",
    "V2": "Your contribution this round: {D}. A value of {D} is shown by default; select whatever number you prefer.",
    "V3": "Your contribution this round: {D}. The system suggests {D}; choose your preferred number.",
}
VARIANTS = list(VARIANT_TEMPLATES.keys())

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "default_experiment")
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
    for tok, lp, alts in positions[:1]:  # only position 0 needed: {0,2,4} are single tokens
        out.append({
            "token": tok,
            "logprob": lp,
            "top_logprobs": [{"token": a_tok, "logprob": a_lp} for a_tok, a_lp in alts],
        })
    return out


def load_model_config(model_id: str) -> tuple[str, dict]:
    with open(CPRD_DATA, encoding="utf-8") as f:
        info = json.load(f)[model_id]
    return info["provider"] + "/" + info["model"], info["kwargs"]


def build_messages(system: str, instructions: str, default_line: str | None) -> list[dict]:
    parts = [instructions] + ([default_line] if default_line else [])
    return construct_messages(system, "\n\n".join(p.strip() for p in parts))


def run_cell(llm: CachedLLM, kwargs: dict, messages: list[dict]) -> dict:
    try:
        response = llm.complete(messages=messages, sample_idx=0, **kwargs)
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    choice = response.choices[0]
    content_text = choice.message.content
    if content_text not in CHOICES:
        return {"error": f"invalid answer {content_text!r} (expected one of {CHOICES})"}

    try:
        positions = extract_positions(choice)
    except RuntimeError as e:
        return {"error": str(e)}

    probs = _reconstruct_number_probs(content_text, positions, CHOICES)
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


def save(model_id: str, condition_suffix: str, payload: dict):
    fname = f"{model_id}-{TASK}-{condition_suffix}-v{VERSION}-n1-{SYSTEM_TYPE}.json"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-instructions-v{VERSION}.txt"), encoding="utf-8") as f:
        instructions = f.read()

    # --- SANITY GATE ---
    check_line = VARIANT_TEMPLATES["V1"].format(D=0)
    check_messages = build_messages(system, instructions, check_line)
    check_user_prompt = check_messages[1]["content"]
    print("=== SANITY GATE: gpt_5.4, anchor=0, V1, full user-prompt string ===")
    print(check_user_prompt)
    print("=== END PROMPT ===")
    ok_leadin = "Your contribution this round:" in check_user_prompt
    ok_choices = "0, 2, or 4" in check_user_prompt or "{0, 2, 4}" in check_user_prompt
    if not ok_leadin or not ok_choices:
        print(f"STOP: lead-in present={ok_leadin}, choice restriction present={ok_choices}. Aborting.")
        return
    print("Lead-in and {0,2,4} restriction confirmed present. Proceeding.\n")

    rows = []  # model, condition(baseline/anchor), variant, spike, argmax, argmax_prob, coverage, error

    for model_id in MODELS:
        model, kwargs = load_model_config(model_id)
        llm = CachedLLM(model=model, db_path=os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "llm_cache.db"),
                        memory_cache=True)

        # --- baseline: no default line at all ---
        print(f"Running {model_id} baseline...")
        messages = build_messages(system, instructions, None)
        result = run_cell(llm, kwargs, messages)
        payload = {"model": model_id, "condition": "baseline", "variant": None}
        if "error" in result:
            payload["error"] = result["error"]
            print(f"  FAILED: {result['error']}")
            rows.append([model_id, "baseline", "", "", "", "", "", result["error"]])
        else:
            payload["crd"] = [{"sample-0": result["probs"]}]
            payload["crd_raw_logprobs"] = [{"sample-0": {
                "content": result["content"], "positions": result["positions_raw"],
            }}]
            payload["coverage"] = result["coverage"]
            payload["argmax"] = result["argmax"]
            payload["argmax_prob"] = result["argmax_prob"]
            print(f"  argmax={result['argmax']} ({result['argmax_prob']:.3f})  coverage={result['coverage']:.3f}")
            rows.append([model_id, "baseline", "", "", result["argmax"], result["argmax_prob"], result["coverage"], ""])
        save(model_id, "baseline", payload)

        # --- 3 anchors x 4 variants ---
        for d in ANCHORS:
            for variant in VARIANTS:
                default_line = VARIANT_TEMPLATES[variant].format(D=d)
                messages = build_messages(system, instructions, default_line)

                print(f"Running {model_id} default={d} {variant}...")
                result = run_cell(llm, kwargs, messages)
                payload = {
                    "model": model_id, "condition": f"default={d}", "variant": variant,
                    "default_line": default_line,
                }
                if "error" in result:
                    payload["error"] = result["error"]
                    print(f"  FAILED: {result['error']}")
                    rows.append([model_id, d, variant, "", "", "", "", result["error"]])
                else:
                    payload["crd"] = [{"sample-0": result["probs"]}]
                    payload["crd_raw_logprobs"] = [{"sample-0": {
                        "content": result["content"], "positions": result["positions_raw"],
                    }}]
                    payload["coverage"] = result["coverage"]
                    payload["argmax"] = result["argmax"]
                    payload["argmax_prob"] = result["argmax_prob"]
                    spike = result["probs"].get(str(d))
                    print(f"  spike@{d}={spike!r}  argmax={result['argmax']} ({result['argmax_prob']:.3f})  "
                         f"coverage={result['coverage']:.3f}")
                    rows.append([model_id, d, variant, spike, result["argmax"], result["argmax_prob"], result["coverage"], ""])
                save(model_id, f"default={d}-{variant}", payload)

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "variant", "spike", "argmax", "argmax_prob", "coverage", "error"])
        w.writerows(rows)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
