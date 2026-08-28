"""Recover CRD no-default baseline probabilities over {0,2,4} for gpt_5.4 and
gpt_5.4_mini, whose original baseline calls didn't have all three tokens in
their top-5 alternatives (both are near-deterministic on "0" at baseline, so
the provider returned fewer than 5 alternatives - not a truncation bug on our
end, the model just puts almost no mass elsewhere).

Two calls only. If a token is still absent from top-5 after the rerun, its
base_p is attributed as the residual (1 - sum of observed valid-token probs)
rather than left unknown, and flagged.
"""
import json
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
MODELS = ["gpt_5.4", "gpt_5.4_mini"]
CHOICES = ["0", "2", "4"]
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "default_experiment")


def extract_positions(choice):
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        raise RuntimeError("No logprobs returned")
    if lp.content is None:
        return [
            (str(tok), None, [(str(t), v) for t, v in lp.top_logprobs[i].items()])
            for i, tok in enumerate(lp.tokens)
        ]
    return [
        (str(t.token), t.logprob, [(str(a.token), a.logprob) for a in t.top_logprobs])
        for t in lp.content
    ]


def positions_to_json(positions):
    out = []
    for tok, lp, alts in positions[:1]:
        out.append({
            "token": tok, "logprob": lp,
            "top_logprobs": [{"token": a_t, "logprob": a_l} for a_t, a_l in alts],
        })
    return out


def load_model_config(model_id):
    with open(CPRD_DATA, encoding="utf-8") as f:
        info = json.load(f)[model_id]
    return info["provider"] + "/" + info["model"], info["kwargs"]


def main():
    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-instructions-v{VERSION}.txt"), encoding="utf-8") as f:
        instructions = f.read()
    messages = construct_messages(system, instructions.strip())

    for model_id in MODELS:
        model, kwargs = load_model_config(model_id)
        # force a fresh, non-cached call: same params, bump sample_idx so the
        # cache key differs from the original baseline call.
        llm = CachedLLM(model=model, db_path=os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "llm_cache.db"),
                        memory_cache=True)

        print(f"Running {model_id} CRD baseline recovery...")
        response = llm.complete(messages=messages, sample_idx=999, **kwargs)
        choice = response.choices[0]
        content_text = choice.message.content
        if content_text not in CHOICES:
            print(f"  FAILED: invalid answer {content_text!r}")
            continue

        positions = extract_positions(choice)
        probs = _reconstruct_number_probs(content_text, positions, CHOICES)

        observed = {k: v for k, v in probs.items() if v is not None}
        missing = [k for k in CHOICES if probs.get(k) is None]
        residual_flag = {}
        if missing:
            residual = max(0.0, 1.0 - sum(observed.values()))
            if len(missing) == 1:
                probs[missing[0]] = residual
                residual_flag[missing[0]] = True
                print(f"  token {missing[0]!r} absent from top-5 even after rerun; "
                     f"attributing residual mass {residual:.6f} to it")
            else:
                # more than one still missing: split residual evenly, flag both
                for tok in missing:
                    probs[tok] = residual / len(missing)
                    residual_flag[tok] = True
                print(f"  tokens {missing} absent from top-5 even after rerun; "
                     f"splitting residual mass {residual:.6f} evenly, flagged")

        coverage = sum(probs.values())
        print(f"  recovered baseline: p0={probs['0']:.6f} p2={probs['2']:.6f} p4={probs['4']:.6f} "
             f"coverage={coverage:.6f}")

        payload = {
            "model": model_id, "condition": "baseline", "variant": None,
            "crd": [{"sample-0": probs}],
            "crd_raw_logprobs": [{"sample-0": {"content": content_text, "positions": positions_to_json(positions)}}],
            "coverage": coverage,
            "argmax": max(probs, key=probs.get),
            "argmax_prob": max(probs.values()),
            "residual_attributed": residual_flag,
            "note": "Baseline recovery rerun (recover_crd_baseline.py): original baseline call "
                   "didn't have all of {0,2,4} in its top-5 alternatives.",
        }
        path = os.path.join(OUTPUT_DIR, f"{model_id}-crd-baseline-v0-n1-blinded.json")
        # keep the original alongside, don't silently clobber
        backup_dir = os.path.join(OUTPUT_DIR, "_baseline_pre_recovery")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{model_id}-crd-baseline-v0-n1-blinded.json")
        if os.path.exists(path) and not os.path.exists(backup_path):
            os.replace(path, backup_path)
            print(f"  backed up original baseline JSON to {backup_path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
