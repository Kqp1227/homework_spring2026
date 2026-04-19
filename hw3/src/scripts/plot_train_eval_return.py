import argparse
import csv
import os

import matplotlib.pyplot as plt


def moving_average(values, window_size):
    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        window = values[start : idx + 1]
        averaged.append(sum(window) / len(window))
    return averaged


def load_series(log_path: str):
    train_steps = []
    train_returns = []
    eval_steps = []
    eval_returns = []

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row.get("step", "").strip()
            if not step:
                continue

            step_value = float(step)

            train_return = row.get("Train_EpisodeReturn", "").strip()
            if train_return:
                train_steps.append(step_value)
                train_returns.append(float(train_return))

            eval_return = row.get("Eval_AverageReturn", "").strip()
            if eval_return:
                eval_steps.append(step_value)
                eval_returns.append(float(eval_return))

    if not train_steps:
        raise ValueError(f"No Train_EpisodeReturn entries found in {log_path}")
    if not eval_steps:
        raise ValueError(f"No Eval_AverageReturn entries found in {log_path}")

    return train_steps, train_returns, eval_steps, eval_returns


def default_output_path(log_path: str) -> str:
    hw3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(log_path)))
    return os.path.join(hw3_root, f"{run_dir}_train_vs_eval_return.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--train_window", type=int, default=20)
    parser.add_argument(
        "--title",
        type=str,
        default="Training Return vs Eval Return",
    )
    args = parser.parse_args()

    train_steps, train_returns, eval_steps, eval_returns = load_series(args.log_path)
    smoothed_train_returns = moving_average(train_returns, args.train_window)
    output_path = args.output or default_output_path(args.log_path)

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        train_steps,
        smoothed_train_returns,
        linewidth=2,
        label=f"Train Return ({args.train_window}-episode avg)",
    )
    plt.plot(eval_steps, eval_returns, linewidth=2, label="Eval Return")
    plt.xlabel("Environment Steps")
    plt.ylabel("Return")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(output_path)


if __name__ == "__main__":
    main()
