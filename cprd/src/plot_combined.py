import argparse
import json
import os

from utilsOutput import plot_combined_models


def main():
    parser = argparse.ArgumentParser(
        description="Overlay every model's CPR extraction curve for one experimental "
                    "condition (default/group_ext/payoff) on a single plot, colored by model."
    )

    parser.add_argument("--task", type=str, default="cpr", help="Name of the task (only 'cpr' is supported)")
    parser.add_argument("--version", type=str, required=True, help="Version of the prompt")
    parser.add_argument("--nsample", type=int, required=True, help="number of samples")
    parser.add_argument("--framing", action="store_true", help="framing flag")
    parser.add_argument("--blinding", action="store_true", help="blinding flag")
    parser.add_argument("--default", type=int, help="Default value")
    parser.add_argument("--group_ext", type=int, help="Group extraction value")
    parser.add_argument("--payoff", type=int, help="Payoff value (used together with --group_ext)")
    parser.add_argument("--output_subdir", type=str, default="", help="Subfolder of output/ the JSON files were saved into")

    args = parser.parse_args()

    if args.blinding:
        system_type = "blinded"
    else:
        system_type = "unblinded"
        if args.framing:
            system_type += "-framed"

    task_suffix = args.task
    if args.default is not None:
        task_suffix = task_suffix + "-default=" + str(args.default)
    if args.group_ext is not None and args.payoff is not None:
        task_suffix = task_suffix + f"-group_ext={args.group_ext}-payoff={args.payoff}"

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models_path = os.path.join(project_dir, "data", "models.json")
    with open(models_path, encoding="utf-8") as f:
        model_list = json.load(f)
    all_model_ids = [k for k in model_list if not k.startswith("_")]

    condition_dir = os.path.join(project_dir, "output", args.output_subdir)
    images_dir = os.path.join(condition_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_path = plot_combined_models(
        condition_dir=condition_dir,
        task_suffix=task_suffix,
        version=args.version,
        nsample=args.nsample,
        system_type=system_type,
        images_dir=images_dir,
        all_model_ids=all_model_ids,
        default_value=args.default,
        group_ext_value=args.group_ext,
        payoff_value=args.payoff,
    )

    if image_path is None:
        print(f"No output files matching '*-{task_suffix}-v{args.version}-n{args.nsample}-{system_type}.json' "
             f"found in {condition_dir}; skipping combined plot.")
    else:
        print(f"Saved combined plot to: {image_path}")


if __name__ == "__main__":
    main()
