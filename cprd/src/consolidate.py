import argparse
import glob
import json
import os

import pandas as pd

from utilsOutput import condition_label, load_run_row, parse_run_filename

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_dir(input_dir: str) -> list[dict]:
    """Parse and load every output JSON under input_dir into flat rows: one
    row per run, with one column per choice value plus a weighted average
    and a notes column flagging anything worth a human's attention.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)):
        info = parse_run_filename(os.path.basename(path))
        if info is None:
            continue
        choices, averages, notes = load_run_row(path, info["task"])
        row = {
            "model": info["model_id"],
            "task": info["task"],
            "condition": info["condition"],
            "version": info["version"],
            "nsample": info["nsample"],
            "system_type": info["system_type"],
        }
        for choice, avg in zip(choices, averages):
            row[f"p_{choice}"] = avg
        if choices:
            row["average"] = sum(int(c) * a for c, a in zip(choices, averages) if c.lstrip("-").isdigit())
        row["notes"] = notes
        row["file"] = os.path.relpath(path, PROJECT_DIR)
        rows.append(row)
    return rows


def add_missing_models(rows: list[dict],
                       models_path: str,
                       task: str,
                       task_suffix: str,
                       version: str,
                       nsample: str,
                       system_type: str,
                       ) -> list[dict]:
    """Add one placeholder row (flagged in `notes`) for every model in
    models.json that has no output file for this exact condition, so failed
    or not-yet-run models still show up in the table instead of silently
    disappearing.
    """
    with open(models_path, encoding="utf-8") as f:
        model_list = json.load(f)
    all_model_ids = [k for k in model_list if not k.startswith("_")]

    target_condition = condition_label(task_suffix, task)
    present = {
        r["model"] for r in rows
        if r["task"] == task and r["version"] == version and r["nsample"] == nsample
        and r["system_type"] == system_type and r["condition"] == target_condition
    }
    for model_id in all_model_ids:
        if model_id not in present:
            rows.append({
                "model": model_id,
                "task": task,
                "condition": target_condition,
                "version": version,
                "nsample": nsample,
                "system_type": system_type,
                "notes": "no output file found (run failed or not yet run)",
                "file": "",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate every output JSON under a folder into one CSV: one row per "
                    "(model, condition, version, nsample, system_type), one column per choice "
                    "value, plus a weighted-average column and a notes column flagging failed "
                    "or partial runs."
    )
    parser.add_argument("--input_dir", type=str, default="output",
                        help="Folder to scan recursively for output JSON files, relative to the "
                             "project root unless an absolute path is given (default: 'output')")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV path to write (default: <input_dir>/consolidated.csv)")
    parser.add_argument("--task", type=str, default="cpr", help="Name of the task (only 'cpr' is supported)")

    # Optional: if --version and --nsample are given, also add a row for
    # every model in models.json that has no matching output file for this
    # exact condition, flagged as failed/missing in the notes column. Meant
    # to be called right after run_all_models.sh, which knows the full
    # condition and the full model list.
    parser.add_argument("--version", type=str, default=None, help="Prompt version of the condition just run")
    parser.add_argument("--nsample", type=str, default=None, help="nsample of the condition just run")
    parser.add_argument("--blinding", action="store_true")
    parser.add_argument("--framing", action="store_true")
    parser.add_argument("--default", type=int, default=None)
    parser.add_argument("--group_ext", type=int, default=None)
    parser.add_argument("--payoff", type=int, default=None)
    parser.add_argument("--models_path", type=str, default=os.path.join(PROJECT_DIR, "data", "models.json"))

    args = parser.parse_args()

    input_dir = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(PROJECT_DIR, args.input_dir)
    rows = scan_dir(input_dir)

    if args.version is not None and args.nsample is not None:
        system_type = "blinded" if args.blinding else ("unblinded-framed" if args.framing else "unblinded")
        task_suffix = args.task
        if args.default is not None:
            task_suffix += f"-default={args.default}"
        if args.group_ext is not None and args.payoff is not None:
            task_suffix += f"-group_ext={args.group_ext}-payoff={args.payoff}"
        rows = add_missing_models(rows, args.models_path, args.task, task_suffix,
                                  args.version, args.nsample, system_type)

    if not rows:
        print(f"No matching output files found under {input_dir}")
        return

    df = pd.DataFrame(rows)
    fixed_cols = ["model", "task", "condition", "version", "nsample", "system_type"]
    choice_cols = sorted((c for c in df.columns if c.startswith("p_")), key=lambda c: int(c[2:]))
    tail_cols = ["average", "notes", "file"]
    df = df[[c for c in fixed_cols + choice_cols + tail_cols if c in df.columns]]
    df = df.sort_values(["model", "condition"]).reset_index(drop=True)

    output_path = args.output or os.path.join(input_dir, "consolidated.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved consolidated table ({len(df)} rows) to: {output_path}")


if __name__ == "__main__":
    main()
