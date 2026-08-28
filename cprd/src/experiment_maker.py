import argparse
import os
import subprocess
import sys

A = 2.3
B = 0.025


def compute_self_payoff(xi: float, group_extraction: float) -> float:
    """pi_i = xi * (a - b * X), where X is the total group extraction."""
    return xi * (A - B * group_extraction)


def run_main(project_dir: str,
            output_subdir: str,
            **main_args,
            ) -> None:
    main_path = os.path.join(project_dir, "src", "main.py")
    cmd = [sys.executable, main_path]
    for key, value in main_args.items():
        if value is False or value is None:
            continue
        flag = f"--{key}"
        if value is True:
            cmd.append(flag)
        else:
            cmd.append(f"{flag}={value}")
    cmd.append(f"--output_subdir={output_subdir}")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_group_extraction_experiment(project_dir: str,
                                    model: str,
                                    version: str,
                                    nsample: int,
                                    xi: float,
                                    default_val: int,
                                    blinding: bool = False,
                                    framing: bool = False,
                                    start: int = 55,
                                    end: int = 100,
                                    step: int = 5,
                                    ) -> None:
    for group_ext in range(start, end + 1, step):
        payoff = compute_self_payoff(xi, group_ext)
        print(f"group_ext={group_ext}  xi={xi}  payoff={payoff}")

        run_main(
            project_dir,
            output_subdir="group_ext",
            model=model,
            task="cpr",
            version=version,
            nsample=nsample,
            blinding=blinding,
            framing=framing,
            group_ext=group_ext,
            payoff=round(payoff),
            default=default_val
        )


def main():
    parser = argparse.ArgumentParser(description="Run batches of CPR experiments over main.py")

    parser.add_argument("--model", type=str, required=True, help="Name of the LLM")
    parser.add_argument("--version", type=str, required=True, help="Version of the prompt")
    parser.add_argument("--nsample", type=int, required=True, help="number of samples")
    parser.add_argument("--framing", action="store_true", help="framing flag")
    parser.add_argument("--blinding", action="store_true", help="blinding flag")
    parser.add_argument("--xi", type=float, required=True, help="Individual's own extraction, used to compute the self payoff")
    parser.add_argument("--start", type=int, default=10, help="Starting group extraction value")
    parser.add_argument("--end", type=int, default=100, help="Ending group extraction value (inclusive)")
    parser.add_argument("--step", type=int, default=5, help="Increment between group extraction values")
    parser.add_argument("--default", type=int, help="Default value")

    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_group_extraction_experiment(
        project_dir,
        model=args.model,
        version=args.version,
        nsample=args.nsample,
        xi=args.xi,
        blinding=args.blinding,
        framing=args.framing,
        start=args.start,
        end=args.end,
        step=args.step,
        default_val=args.default
    )


if __name__ == "__main__":
    main()
