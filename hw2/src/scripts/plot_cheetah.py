import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_halfcheetah_runs(exp_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load all HalfCheetah-v4 runs under exp/, returning (label, df)."""
    runs: List[Tuple[str, pd.DataFrame]] = []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("HalfCheetah-v4_"):
            continue
        csv_path = p / "log.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Label: everything after env name, before seed/timestamp.
        # Example: HalfCheetah-v4_cheetah_baseline_sd1_2026... -> cheetah_baseline
        name_no_env = p.name.split("HalfCheetah-v4_", 1)[-1]
        label = name_no_env.split("_sd", 1)[0]
        runs.append((label, df))
    return runs


def plot_baseline_loss(runs: List[Tuple[str, pd.DataFrame]], out_path: Path):
    """Plot Baseline Loss vs Train_EnvstepsSoFar for all runs that have a baseline."""
    plt.figure(figsize=(8, 5))
    has_any = False
    for label, df in runs:
        if "Baseline Loss" not in df.columns:
            continue
        if "Train_EnvstepsSoFar" not in df.columns:
            continue
        x = df["Train_EnvstepsSoFar"].astype(float).to_numpy()
        y = df["Baseline Loss"].astype(float).to_numpy()
        plt.plot(x, y, label=label)
        has_any = True

    if not has_any:
        print("No runs with 'Baseline Loss' found; skipping baseline-loss plot.")
        plt.close()
        return

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Baseline Loss")
    plt.title("HalfCheetah-v4: baseline loss vs env steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def plot_eval_return(runs: List[Tuple[str, pd.DataFrame]], out_path: Path):
    """Plot Eval_AverageReturn vs Train_EnvstepsSoFar for all HalfCheetah runs."""
    plt.figure(figsize=(8, 5))
    has_any = False
    for label, df in runs:
        if "Eval_AverageReturn" not in df.columns:
            continue
        if "Train_EnvstepsSoFar" not in df.columns:
            continue
        x = df["Train_EnvstepsSoFar"].astype(float).to_numpy()
        y = df["Eval_AverageReturn"].astype(float).to_numpy()
        plt.plot(x, y, label=label)
        has_any = True

    if not has_any:
        print("No runs with 'Eval_AverageReturn' found; skipping eval-return plot.")
        plt.close()
        return

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.title("HalfCheetah-v4: eval return vs env steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="Directory containing HalfCheetah-v4_* run subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Where to save the generated plots.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)

    runs = _load_halfcheetah_runs(exp_dir)
    if not runs:
        print(f"No HalfCheetah-v4 runs found under {exp_dir}")
        return

    plot_baseline_loss(runs, out_dir / "cheetah_baseline_loss.png")
    plot_eval_return(runs, out_dir / "cheetah_eval_return.png")


if __name__ == "__main__":
    main()

