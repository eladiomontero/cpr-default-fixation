import argparse
import os

from utilsOutput import plot_model_conditions


def main():
    parser = argparse.ArgumentParser(
        description="Overlay every experimental condition run so far for a single model "
                    "(plain / default=.../group_ext=.../payoff=...) on one plot, colored by "
                    "condition. Scans output/ recursively, so it works on whatever has been "
                    "generated so far, regardless of which output_subdir it landed in and "
                    "independent of run_all_models.sh."
    )

    parser.add_argument("--model", type=str, required=True, help="Model id, as keyed in data/models.json")
    parser.add_argument("--task", type=str, default="cpr", help="Name of the task (only 'cpr' is supported)")
    parser.add_argument("--input_dir", type=str, default="output",
                        help="Folder to search recursively for output JSON files, relative to the "
                             "project root unless an absolute path is given (default: 'output')")
    parser.add_argument("--version", type=str, help="Restrict to this prompt version (default: any)")
    parser.add_argument("--blinding", action="store_true", help="Restrict to blinded runs")
    parser.add_argument("--framing", action="store_true", help="Restrict to unblinded-framed runs (only used without --blinding)")

    args = parser.parse_args()

    system_type = None
    if args.blinding:
        system_type = "blinded"
    elif args.framing:
        system_type = "unblinded-framed"

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_root = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(project_dir, args.input_dir)
    images_dir = os.path.join(output_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_path = plot_model_conditions(
        model_id=args.model,
        task=args.task,
        output_root=output_root,
        images_dir=images_dir,
        version=args.version,
        system_type=system_type,
    )

    if image_path is None:
        print(f"No output files found for model '{args.model}' under {output_root}; skipping plot.")
    else:
        print(f"Saved plot to: {image_path}")


if __name__ == "__main__":
    main()
