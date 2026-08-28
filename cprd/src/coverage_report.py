import argparse
import glob
import json
import os

import pandas as pd

from utilsOutput import parse_run_filename

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLAG_THRESHOLD = 0.95  # below this: flag, exclude from the spike metric
DROP_THRESHOLD = 0.90  # below this: drop from the mechanistic set entirely


def sample_coverage(sample: dict) -> float:
    """Raw summed probability of the returned top-k tokens that fell within
    the valid answer set (0-30) at the answer position - i.e. the choices
    that ended up non-None in get_probs_task's output. Not renormalised;
    that's a separate step downstream once a model-condition clears the
    coverage threshold.
    """
    return sum(v for v in sample.values() if v is not None)


def classify(coverage: float) -> str:
    if coverage < DROP_THRESHOLD:
        return "DROP (mechanistic set)"
    if coverage < FLAG_THRESHOLD:
        return "FLAG (exclude from spike metric)"
    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Compute top-k answer-token coverage (raw summed probability of choices "
                    "0-30 among the returned top-k logprobs) per model per condition, and "
                    "classify against the 0.95 (flag) / 0.90 (drop) thresholds."
    )
    parser.add_argument("--input_dir", type=str, default="output",
                        help="Folder to scan recursively for output JSON files")
    parser.add_argument("--task", type=str, default="cpr")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV path to write (default: <input_dir>/coverage_report.csv)")
    args = parser.parse_args()

    input_dir = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(PROJECT_DIR, args.input_dir)

    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)):
        info = parse_run_filename(os.path.basename(path))
        if info is None or info["task"] != args.task:
            continue

        try:
            with open(path, encoding="utf-8") as f:
                output = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            rows.append({"model": info["model_id"], "condition": info["condition"],
                        "version": info["version"], "system_type": info["system_type"],
                        "sample": None, "coverage": None, "status": "ERROR", "notes": f"unreadable file: {e}"})
            continue

        samples = output.get(args.task, [{}])[0]
        if not samples:
            rows.append({"model": info["model_id"], "condition": info["condition"],
                        "version": info["version"], "system_type": info["system_type"],
                        "sample": None, "coverage": None, "status": "ERROR", "notes": "no valid samples"})
            continue

        for sample_key, sample in samples.items():
            coverage = sample_coverage(sample)
            rows.append({
                "model": info["model_id"],
                "condition": info["condition"],
                "version": info["version"],
                "system_type": info["system_type"],
                "sample": sample_key,
                "coverage": coverage,
                "status": classify(coverage),
                "notes": "",
                "file": os.path.relpath(path, PROJECT_DIR),
            })

    if not rows:
        print(f"No matching '{args.task}' output files found under {input_dir}")
        return

    df = pd.DataFrame(rows)
    output_path = args.output or os.path.join(input_dir, "coverage_report.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved coverage report ({len(df)} rows) to: {output_path}")

    flagged = df[df["status"] != "ok"]
    if not flagged.empty:
        print(f"\n{len(flagged)} model-condition-sample(s) below the {FLAG_THRESHOLD} coverage threshold:")
        print(flagged[["model", "condition", "coverage", "status"]].to_string(index=False))
    else:
        print(f"\nAll model-condition-samples clear the {FLAG_THRESHOLD} coverage threshold.")


if __name__ == "__main__":
    main()
