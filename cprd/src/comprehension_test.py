import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm

from utilsLiteLLM import CachedLLM
from utilsOutput import construct_messages

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mirrors the 5 questions in prompts/cpr/comprehension-test-v0.txt (taken
# verbatim from the "Comprehension test for task 3" page of the experiment
# instructions PDF). Each answer_fn takes the {group_extraction: reward_per_point}
# table parsed from cpr-table.txt and returns the expected numeric answer,
# so the expected answers stay in sync with the table actually shown to the
# model rather than being hardcoded separately from it. Verified against the
# payoff formula in journal.pone.0331348 (pi_i = x_i * (a - b*X), a=2.3,
# b=0.025) - cpr-table.txt matches the paper exactly; it's the PDF's own
# sandbox screenshot table that's off by a stray x10, not this one.
QUESTIONS = [
    {"num": 1, "answer_fn": lambda table: 15 + 45},
    {"num": 2, "answer_fn": lambda table: 20 - 5},
    {"num": 3, "answer_fn": lambda table: 10 * table[60]},
    {"num": 4, "answer_fn": lambda table: 10 * table[40]},
    {"num": 5, "answer_fn": lambda table: 2 * table[60]},
]

ANSWER_LINE_RE = re.compile(r"^\s*(\d)\s*[:.\)]\s*(-?\d+(?:\.\d+)?)")
ANY_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_reward_table(path: str) -> dict[int, float]:
    table = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            extraction, _reward, per_point = line.split(";")
            table[int(extraction)] = float(per_point)
    return table


def parse_answers(response_text: str) -> dict[int, float]:
    """Parse a model's response into {question_num: value}, preferring the
    requested 'N: value' line format but falling back to reading off numbers
    in order if the model didn't follow it exactly.
    """
    answers = {}
    for line in response_text.splitlines():
        m = ANSWER_LINE_RE.match(line)
        if m:
            answers[int(m.group(1))] = float(m.group(2))

    if len(answers) < len(QUESTIONS):
        fallback_numbers = ANY_NUMBER_RE.findall(response_text)
        for q, num_str in zip((q["num"] for q in QUESTIONS if q["num"] not in answers), fallback_numbers):
            answers.setdefault(q, float(num_str))

    return answers


def run_generation(llm: CachedLLM,
                   messages: list[dict],
                   call_kwargs: dict,
                   gen_idx: int,
                   expected: dict[int, float],
                   ) -> dict:
    """Make one generation call and grade it. Distinct gen_idx values (used
    as sample_idx) give distinct cache keys, so each of the N generations is
    an independent draw from the model rather than N cache hits on the same
    completion.
    """
    row = {"generation": gen_idx}
    try:
        response = llm.complete(messages=messages, sample_idx=gen_idx, **call_kwargs)
        response_text = response.choices[0].message.content or ""
    except Exception as e:
        row["notes"] = f"call failed: {e}"
        return row

    row["raw_response"] = response_text
    parsed = parse_answers(response_text)
    num_correct = 0
    for q in QUESTIONS:
        n = q["num"]
        given = parsed.get(n)
        row[f"q{n}_answer"] = given
        row[f"q{n}_correct"] = given is not None and abs(given - expected[n]) < 1e-6
        num_correct += row[f"q{n}_correct"]
    row["fraction_correct"] = num_correct / len(QUESTIONS)
    row["notes"] = "" if len(parsed) == len(QUESTIONS) else f"only parsed {len(parsed)}/{len(QUESTIONS)} answers"
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run the CPR comprehension test (the 5 questions from the experiment "
                    "instructions PDF) N times per model as independent draws, and report "
                    "each model's mean fraction correct (0-1) as a continuous comprehension "
                    "score - cheap since each draw is one short prompt, five short answers, "
                    "no group interaction."
    )
    parser.add_argument("--version", type=str, default="0", help="Version of the comprehension-test prompt")
    parser.add_argument("--n_generations", type=int, default=20, help="Independent draws per model (10-20 recommended)")
    parser.add_argument("--models_path", type=str, default=os.path.join(PROJECT_DIR, "data", "models.json"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_DIR, "output", "comprehension_test.csv"),
                        help="Per-model summary CSV (mean fraction correct etc.)")
    parser.add_argument("--output_generations", type=str, default=None,
                        help="Per-generation detail CSV (default: <output dir>/comprehension_test_generations.csv)")
    parser.add_argument("--db_path", type=str, default=os.path.join(PROJECT_DIR, "llm_cache.db"))

    args = parser.parse_args()

    prompts_dir = os.path.join(PROJECT_DIR, "prompts")

    with open(os.path.join(prompts_dir, "system", "comprehension-system-prompt.txt"), encoding="utf-8") as f:
        system_prompt = f.read()
    with open(os.path.join(prompts_dir, "cpr", "cpr-instructions-v0.txt"), encoding="utf-8") as f:
        cpr_instructions = f.read()
    with open(os.path.join(prompts_dir, "cpr", "cpr-table.txt"), encoding="utf-8") as f:
        cpr_table_text = f.read()
    with open(os.path.join(prompts_dir, "cpr", f"comprehension-test-v{args.version}.txt"), encoding="utf-8") as f:
        comprehension_text = f.read()

    reward_table = load_reward_table(os.path.join(prompts_dir, "cpr", "cpr-table.txt"))
    expected = {q["num"]: q["answer_fn"](reward_table) for q in QUESTIONS}

    user_prompt = cpr_instructions + "\n" + cpr_table_text + "\n" + comprehension_text
    messages = construct_messages(system_prompt, user_prompt)

    with open(args.models_path, encoding="utf-8") as f:
        model_list = json.load(f)
    model_ids = [k for k in model_list if not k.startswith("_")]

    generation_rows = []
    for model_id in tqdm(model_ids):
        model_info = model_list[model_id]
        model = model_info["provider"] + "/" + model_info["model"]
        call_kwargs = {k: v for k, v in model_info["kwargs"].items() if k not in ("logprobs", "top_logprobs", "n")}
        llm = CachedLLM(model=model, db_path=args.db_path, memory_cache=True)

        for gen_idx in range(args.n_generations):
            row = run_generation(llm, messages, call_kwargs, gen_idx, expected)
            row["model"] = model_id
            generation_rows.append(row)

    gen_df = pd.DataFrame(generation_rows)
    fixed_cols = ["model", "generation"]
    per_q_cols = [c for n in range(1, len(QUESTIONS) + 1) for c in (f"q{n}_answer", f"q{n}_correct")]
    tail_cols = ["fraction_correct", "notes", "raw_response"]
    gen_df = gen_df[[c for c in fixed_cols + per_q_cols + tail_cols if c in gen_df.columns]]

    output_generations = args.output_generations or os.path.join(
        os.path.dirname(args.output), "comprehension_test_generations.csv")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    gen_df.to_csv(output_generations, index=False)
    print(f"Saved per-generation results ({len(gen_df)} rows) to: {output_generations}")

    failed = gen_df["fraction_correct"].isna() if "fraction_correct" in gen_df.columns else pd.Series(dtype=bool)
    summary_rows = []
    for model_id, group in gen_df.groupby("model"):
        ok = group.dropna(subset=["fraction_correct"]) if "fraction_correct" in group.columns else group.iloc[0:0]
        row = {
            "model": model_id,
            "n_generations": len(group),
            "n_failed_calls": int((group["fraction_correct"].isna() if "fraction_correct" in group.columns else pd.Series([True] * len(group))).sum()),
            "mean_fraction_correct": ok["fraction_correct"].mean() if len(ok) else float("nan"),
            "std_fraction_correct": ok["fraction_correct"].std() if len(ok) > 1 else 0.0,
        }
        for q in QUESTIONS:
            n = q["num"]
            col = f"q{n}_correct"
            row[f"q{n}_accuracy"] = ok[col].mean() if col in ok.columns and len(ok) else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_fraction_correct", ascending=False)
    summary_df.to_csv(args.output, index=False)
    print(f"Saved per-model comprehension scores ({len(summary_df)} models) to: {args.output}")


if __name__ == "__main__":
    main()
