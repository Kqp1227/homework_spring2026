import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_eval_series(log_path: str):
    steps = []
    eval_returns = []

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row.get("step", "").strip()
            eval_return = row.get("Eval_AverageReturn", "").strip()
            if not step or not eval_return:
                continue

            steps.append(float(step))
            eval_returns.append(float(eval_return))

    if not steps:
        raise ValueError(f"No Eval_AverageReturn entries found in {log_path}")

    return steps, eval_returns


def default_output_path() -> str:
    hw3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(hw3_root, "lunarlander_hyperparameter_eval_return.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_paths", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--title",
        type=str,
        default="LunarLander-v2 Eval Return vs Environment Steps",
    )
    args = parser.parse_args()

    if len(args.log_paths) != len(args.labels):
        raise ValueError("--log_paths and --labels must have the same length")

    output_path = args.output or default_output_path()

    plt.figure(figsize=(9, 5.5))
    for log_path, label in zip(args.log_paths, args.labels):
        steps, eval_returns = load_eval_series(log_path)
        plt.plot(steps, eval_returns, linewidth=2, label=label)

    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Return")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(output_path)


if __name__ == "__main__":
    main()
