"""Targeted re-run: Qwen3_235B_A22B_Instruct only, all 7 mult_default CPR
conditions (baseline + defaults 0/11/18/23/26/30), with logprobs persisted
raw (not just collapsed) so the reconstruction is reproducible without ever
needing to re-hit the API.

Baseline uses a zero-padded two-digit answer format ("00".."30") so every
answer - including one-digit values 0-9 - spans two tokens and the full
first-digit x second-digit distribution is observable. The six default
conditions keep the normal (unpadded) format used everywhere else in the
pipeline.
"""
import json
import math
import os

from utilsLiteLLM import CachedLLM
from utilsOutput import _reconstruct_number_probs, construct_messages

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen3_235B_A22B_Instruct"
TASK = "cpr"
VERSION = "0"
NSAMPLE = "1"
SYSTEM_TYPE = "blinded"

CHOICES = [str(i) for i in range(0, 31)]
CHOICES_PADDED = [f"{i:02d}" for i in range(0, 31)]

DEFAULTS = [0, 11, 18, 23, 26, 30]


def extract_positions(choice) -> list[tuple[str, float, list[tuple[str, float]]]]:
    """Same branching as utilsOutput.get_probs_task, factored out so we can
    also keep the raw (token, logprob, alts) tuples around for persistence.
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        raise RuntimeError("No logprobs returned - provider/model does not support logprobs")
    if lp.content is None:  # huggingface-style: parallel .tokens / .top_logprobs arrays
        return [
            (str(tok), None, [(str(t), v) for t, v in lp.top_logprobs[i].items()])
            for i, tok in enumerate(lp.tokens)
        ]
    return [
        (str(t.token), t.logprob, [(str(a.token), a.logprob) for a in t.top_logprobs])
        for t in lp.content
    ]


def positions_to_json(positions: list) -> list[dict]:
    """First two positions only (all _reconstruct_number_probs ever uses),
    in a plain-JSON-serializable shape."""
    out = []
    for tok, lp, alts in positions[:2]:
        out.append({
            "token": tok,
            "logprob": lp,
            "top_logprobs": [{"token": a_tok, "logprob": a_lp} for a_tok, a_lp in alts],
        })
    return out


def run_condition(llm: CachedLLM, kwargs: dict, messages: list[dict],
                  padded: bool) -> dict:
    """Call the model once, reconstruct p_0..p_30, and return the full
    output dict (with raw logprobs) ready to json.dump."""
    response = llm.complete(messages=messages, sample_idx=0, **kwargs)
    choice = response.choices[0]
    content_text = choice.message.content

    if content_text not in (CHOICES_PADDED if padded else CHOICES):
        raise RuntimeError(f"Model answer {content_text!r} is not a valid choice "
                           f"(padded={padded})")

    positions = extract_positions(choice)
    choices = CHOICES_PADDED if padded else CHOICES
    probs = _reconstruct_number_probs(content_text, positions, choices)

    if padded:
        # map "00".."30" back onto the canonical "0".."30" keys everywhere
        # else in the pipeline expects (parse_run_filename/load_run_row/
        # consolidate.py all key on plain choice strings).
        probs = {str(i): probs[f"{i:02d}"] for i in range(31)}

    return {
        "cpr": [{"sample-0": probs}],
        "cpr_raw_logprobs": [{"sample-0": {
            "content": content_text,
            "positions": positions_to_json(positions),
        }}],
    }


def main():
    data_dir = os.path.join(PROJECT_DIR, "data")
    with open(os.path.join(data_dir, "models.json"), encoding="utf-8") as f:
        model_info = json.load(f)[MODEL_ID]
    model = model_info["provider"] + "/" + model_info["model"]
    kwargs = model_info["kwargs"]

    llm = CachedLLM(model=model, db_path=os.path.join(PROJECT_DIR, "llm_cache.db"), memory_cache=True)

    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-instructions-v{VERSION}.txt"), encoding="utf-8") as f:
        instructions = f.read()
    with open(os.path.join(prompts_dir, TASK, f"{TASK}-table.txt"), encoding="utf-8") as f:
        cpr_table = f.read()
    with open(os.path.join(prompts_dir, TASK, "default-v0.txt"), encoding="utf-8") as f:
        default_template = f.read()

    padded_addendum = ("\n\nAnswer format: respond with EXACTLY TWO DIGITS representing your "
                       "chosen number (00 to 30), e.g. \"05\" for five, \"20\" for twenty, "
                       "\"00\" for zero. Do not include any other characters.")

    output_dir = os.path.join(PROJECT_DIR, "output", "mult_default")
    # kept OUTSIDE output_dir - consolidate.py globs output_dir recursively,
    # and a backup left inside would get scanned as a second (stale) run.
    backup_dir = os.path.join(PROJECT_DIR, "output", "_stale_qwen_backup")
    os.makedirs(backup_dir, exist_ok=True)

    written = []

    def save(task_suffix: str, output: dict):
        fname = f"{MODEL_ID}-{task_suffix}-v{VERSION}-n{NSAMPLE}-{SYSTEM_TYPE}.json"
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            backup_path = os.path.join(backup_dir, fname)
            if not os.path.exists(backup_path):
                os.replace(path, backup_path)
                print(f"  backed up stale {fname} -> _stale_qwen_backup/")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        written.append(path)
        print(f"  saved {path}")

    # --- baseline, zero-padded ---
    print("Running baseline (zero-padded two-digit format)...")
    messages = construct_messages(system, (instructions + padded_addendum + "\n\n" + cpr_table).strip())
    out = run_condition(llm, kwargs, messages, padded=True)
    save(TASK, out)

    # --- 6 defaults, normal format ---
    for d in DEFAULTS:
        print(f"Running default={d}...")
        default_text = default_template.replace("%i", str(d))
        parts = [instructions, cpr_table, default_text]
        messages = construct_messages(system, "\n\n".join(p.strip() for p in parts))
        out = run_condition(llm, kwargs, messages, padded=False)
        save(f"{TASK}-default={d}", out)

    print("\nFiles written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
