"""CRD comprehension test, mirroring cprd/src/comprehension_test.py's
multi-generation / grade-against-expected pipeline, so the two games'
comprehension scores are directly comparable.

7 questions (certainty condition only - drops the uncertainty-only Q8):
numeric (1,2,3,4,7), a single-letter choice (5), and Yes/No (6). Answers
graded against fixed expected values for the certainty-treatment CRD game
(4 players, 50 ECU private account, {0,2,4} contribution, 10 rounds,
target 80, >=80 keep 100%, <80 keep 10%).

Same blinded system prompt as the CRD main run (not a dedicated
comprehension-system-prompt like CPR uses) - per task spec.
"""
import argparse
import collections
import json
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CRD/
CPRD_SRC = os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "src")
CPRD_DATA = os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "data", "models.json")
sys.path.insert(0, CPRD_SRC)

from utilsLiteLLM import CachedLLM  # noqa: E402
from utilsOutput import construct_messages  # noqa: E402

SYSTEM_TYPE = "blinded"
VERSION = "0"

QUESTIONS = [
    {"num": 1, "type": "numeric", "expected": 4},
    {"num": 2, "type": "numeric", "expected": 50},
    {"num": 3, "type": "numeric", "expected": 10},
    {"num": 4, "type": "numeric", "expected": 100},
    {"num": 5, "type": "choice", "expected": "D"},
    {"num": 6, "type": "yesno", "expected": "No"},
    {"num": 7, "type": "numeric", "expected": 10},
]

ANSWER_LINE_RE = re.compile(r"^\s*(\d)\s*[:.\)]\s*(.+?)\s*$")
ANY_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def grade(qtype: str, raw_value: str, expected):
    """Parse+grade one answer against its expected value. Returns
    (parsed_value_for_logging, is_correct)."""
    text = raw_value.strip()
    if qtype == "numeric":
        m = ANY_NUMBER_RE.search(text)
        if not m:
            return text, False
        val = float(m.group(0))
        return val, abs(val - float(expected)) < 1e-6
    if qtype == "choice":
        norm = text.strip().upper()
        letter = norm[0] if norm else ""
        is_correct = letter == expected or "EVERY GROUP MEMBER" in norm or "EVERY MEMBER" in norm
        return text, bool(is_correct)
    if qtype == "yesno":
        norm = text.strip().lower()
        if norm.startswith("y"):
            parsed = "Yes"
        elif norm.startswith("n"):
            parsed = "No"
        else:
            parsed = text
        return parsed, parsed.lower() == expected.lower()
    raise ValueError(qtype)


def parse_answers(response_text: str) -> dict[int, str]:
    """Parse a model's response into {question_num: raw_value_text},
    preferring the requested 'N: value' line format, falling back to
    reading numbers off in order for any still-missing numeric questions.
    """
    answers = {}
    for line in response_text.splitlines():
        m = ANSWER_LINE_RE.match(line)
        if m:
            answers[int(m.group(1))] = m.group(2)

    missing_numeric = [q["num"] for q in QUESTIONS if q["num"] not in answers and q["type"] == "numeric"]
    if missing_numeric:
        fallback_numbers = ANY_NUMBER_RE.findall(response_text)
        # only use fallback numbers not already consumed by parsed lines
        used = {v for v in answers.values()}
        for q_num, num_str in zip(missing_numeric, fallback_numbers):
            answers.setdefault(q_num, num_str)

    return answers


def run_generation(llm: CachedLLM, messages: list[dict], call_kwargs: dict, gen_idx: int) -> dict:
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
        raw_value = parsed.get(n)
        if raw_value is None:
            row[f"q{n}_answer"] = None
            row[f"q{n}_correct"] = False
            continue
        parsed_val, correct = grade(q["type"], str(raw_value), q["expected"])
        row[f"q{n}_answer"] = parsed_val
        row[f"q{n}_correct"] = correct
        num_correct += correct
    row["fraction_correct"] = num_correct / len(QUESTIONS)
    row["notes"] = "" if len(parsed) == len(QUESTIONS) else f"only parsed {len(parsed)}/{len(QUESTIONS)} answers"
    return row


def main():
    parser = argparse.ArgumentParser(description="Run the CRD comprehension test (7 questions, certainty "
                                                  "condition) N times per model, mirroring the CPR comprehension "
                                                  "test's format so the two games' scores are comparable.")
    parser.add_argument("--n_generations", type=int, default=20)
    parser.add_argument("--models_path", type=str, default=CPRD_DATA)
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_DIR, "output", "default_experiment", "crd_comprehension.csv"))
    parser.add_argument("--output_generations", type=str, default=None)
    parser.add_argument("--db_path", type=str,
                        default=os.path.join(os.path.dirname(PROJECT_DIR), "cprd", "llm_cache.db"))
    args = parser.parse_args()

    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    with open(os.path.join(prompts_dir, "system", f"{SYSTEM_TYPE}-system-prompt.txt"), encoding="utf-8") as f:
        system_prompt = f.read()
    with open(os.path.join(prompts_dir, "crd", f"crd-instructions-comprehension-v{VERSION}.txt"), encoding="utf-8") as f:
        crd_instructions = f.read()
    with open(os.path.join(prompts_dir, "crd", f"crd-comprehension-test-v{VERSION}.txt"), encoding="utf-8") as f:
        comprehension_text = f.read()

    user_prompt = crd_instructions + "\n" + comprehension_text
    messages = construct_messages(system_prompt, user_prompt)

    with open(args.models_path, encoding="utf-8") as f:
        model_list = json.load(f)
    model_ids = [
        "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
        "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
    ]

    generation_rows = []
    for model_id in tqdm(model_ids):
        model_info = model_list[model_id]
        model = model_info["provider"] + "/" + model_info["model"]
        call_kwargs = {k: v for k, v in model_info["kwargs"].items() if k not in ("logprobs", "top_logprobs", "n")}
        llm = CachedLLM(model=model, db_path=args.db_path, memory_cache=True)

        for gen_idx in range(args.n_generations):
            row = run_generation(llm, messages, call_kwargs, gen_idx)
            row["model"] = model_id
            generation_rows.append(row)

    gen_df = pd.DataFrame(generation_rows)
    fixed_cols = ["model", "generation"]
    per_q_cols = [c for n in range(1, len(QUESTIONS) + 1) for c in (f"q{n}_answer", f"q{n}_correct")]
    tail_cols = ["fraction_correct", "notes", "raw_response"]
    gen_df = gen_df[[c for c in fixed_cols + per_q_cols + tail_cols if c in gen_df.columns]]

    output_generations = args.output_generations or os.path.join(
        os.path.dirname(args.output), "crd_comprehension_generations.csv")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    gen_df.to_csv(output_generations, index=False)
    print(f"Saved per-generation results ({len(gen_df)} rows) to: {output_generations}")

    summary_rows = []
    for model_id, group in gen_df.groupby("model"):
        ok = group.dropna(subset=["fraction_correct"]) if "fraction_correct" in group.columns else group.iloc[0:0]
        row = {
            "model": model_id,
            "n_generations": len(group),
            "n_failed_calls": int((group["fraction_correct"].isna() if "fraction_correct" in group.columns
                                  else pd.Series([True] * len(group))).sum()),
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

    print("\n=== PER-MODEL SUMMARY ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:26s} mean_fraction_correct={row['mean_fraction_correct']:.3f} "
             f"(n={row['n_generations']}, failed={row['n_failed_calls']})")

    print("\n=== MOST COMMON WRONG ANSWER PER QUESTION, PER MODEL ===")
    for model_id, group in gen_df.groupby("model"):
        print(f"  -- {model_id} --")
        for q in QUESTIONS:
            n = q["num"]
            wrong = group[group[f"q{n}_correct"] == False][f"q{n}_answer"]  # noqa: E712
            wrong = wrong.dropna()
            if wrong.empty:
                print(f"    Q{n}: no wrong answers (all correct or unparsed)")
                continue
            counts = collections.Counter(str(v) for v in wrong)
            most_common, count = counts.most_common(1)[0]
            print(f"    Q{n} (expected {q['expected']}): most common wrong answer = {most_common!r} ({count}/{len(wrong)} wrong answers)")


if __name__ == "__main__":
    main()
