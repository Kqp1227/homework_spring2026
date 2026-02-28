import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _extract_label(run_dir_name: str) -> str:
    # Example: CartPole-v0_cartpole_rtg_na_sd1_20260223_225734
    prefix = run_dir_name.split("_sd", 1)[0]
    if "_" not in prefix:
        return run_dir_name
    return prefix.split("_", 1)[1]


def _load_runs(exp_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    runs: List[Tuple[str, pd.DataFrame]] = []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir():
            continue
        csv_path = p / "log.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        runs.append((_extract_label(p.name), df))
    return runs


def _select_groups(
    runs: List[Tuple[str, pd.DataFrame]],
) -> Tuple[List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]]]:
    small, large = [], []
    for label, df in runs:
        if not label.startswith("cartpole"):
            continue
        if label.startswith("cartpole_lb"):
            large.append((label, df))
        else:
            small.append((label, df))
    return small, large


def _plot_group(
    runs: List[Tuple[str, pd.DataFrame]],
    out_path: Path,
    y_key: str,
    title: str,
):
    if not runs:
        raise RuntimeError(f"No runs found for plot: {title}")

    plt.figure(figsize=(8, 5))
    for label, df in runs:
        if "Train_EnvstepsSoFar" not in df.columns or y_key not in df.columns:
            continue
        x = df["Train_EnvstepsSoFar"].astype(float).to_numpy()
        y = df[y_key].astype(float).to_numpy()
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="Directory containing run subfolders (each with log.csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Where to write the generated png files.",
    )
    parser.add_argument(
        "--y_key",
        type=str,
        default="Eval_AverageReturn",
        help="Column name to plot on the y-axis (e.g., Eval_AverageReturn or Train_AverageReturn).",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)

    runs = _load_runs(exp_dir)
    small, large = _select_groups(runs)

    _plot_group(
        small,
        out_dir / "cartpole_small_batch.png",
        y_key=args.y_key,
        title="CartPole (small batch): average return vs env steps",
    )
    _plot_group(
        large,
        out_dir / "cartpole_large_batch.png",
        y_key=args.y_key,
        title="CartPole (large batch): average return vs env steps",
    )

    print(f"Wrote: {os.fspath(out_dir / 'cartpole_small_batch.png')}")
    print(f"Wrote: {os.fspath(out_dir / 'cartpole_large_batch.png')}")


if __name__ == "__main__":
    main()

