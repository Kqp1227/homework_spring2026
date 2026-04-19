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


def default_output_path(log_path: str) -> str:
    hw3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(log_path)))
    filename = f"{run_dir}_eval_return.png"
    return os.path.join(hw3_root, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default="Eval Return vs Environment Steps")
    args = parser.parse_args()

    steps, eval_returns = load_eval_series(args.log_path)

    output_path = args.output
    if output_path is None:
        output_path = default_output_path(args.log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, eval_returns, linewidth=2)
    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Return")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(output_path)


if __name__ == "__main__":
    main()
