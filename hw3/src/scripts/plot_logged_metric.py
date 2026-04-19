import argparse
import csv
import os

import matplotlib.pyplot as plt


def moving_average(values, window_size):
    if window_size <= 1:
        return values

    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        window = values[start : idx + 1]
        averaged.append(sum(window) / len(window))
    return averaged


def load_metric_series(log_path: str, metric: str):
    steps = []
    values = []

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row.get("step", "").strip()
            metric_value = row.get(metric, "").strip()
            if not step or not metric_value:
                continue

            steps.append(float(step))
            values.append(float(metric_value))

    if not steps:
        raise ValueError(f"No {metric} entries found in {log_path}")

    return steps, values


def default_output_path(metric: str) -> str:
    hw3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    filename = f"{metric.lower()}_plot.png"
    return os.path.join(hw3_root, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_paths", type=str, nargs="+", required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlabel", type=str, default="Environment Steps")
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--yscale", type=str, default="linear")
    args = parser.parse_args()

    labels = args.labels
    if labels is None:
        labels = [os.path.basename(os.path.dirname(path)) for path in args.log_paths]
    if len(labels) != len(args.log_paths):
        raise ValueError("--labels must match the number of --log_paths")

    output_path = args.output or default_output_path(args.metric)
    title = args.title or f"{args.metric} vs Environment Steps"
    ylabel = args.ylabel or args.metric

    plt.figure(figsize=(9, 5.5))
    for log_path, label in zip(args.log_paths, labels):
        steps, values = load_metric_series(log_path, args.metric)
        values = moving_average(values, args.smooth)
        plt.plot(steps, values, linewidth=2, label=label)

    plt.xlabel(args.xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(args.yscale)
    plt.grid(True, alpha=0.3)
    if len(args.log_paths) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(output_path)


if __name__ == "__main__":
    main()
