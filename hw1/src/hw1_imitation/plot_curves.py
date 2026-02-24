import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ================== 路径 ==================
TRAIN_LOG = Path(
    "/Users/qpkong/Desktop/courses/CS285/homework_spring2026/hw1/wandb_export_2026-02-11T02_46_57.790-08_00.csv"
)
EVAL_LOG = Path("/Users/qpkong/Desktop/courses/CS285/homework_spring2026/hw1/exp/seed_42_20260211_024233/log.csv")

OUT_TRAIN = Path("train_loss_flow.png")
OUT_EVAL = Path("eval_reward_flow.png")
# ==========================================


def plot_train_loss():
    steps = []
    losses = []

    with open(TRAIN_LOG, "r") as f:
        reader = csv.DictReader(f)

        # 自动找到 train/loss 那一列
        loss_key = None
        for k in reader.fieldnames:
            if "train/loss" in k and "__" not in k:
                loss_key = k
                break

        if loss_key is None:
            raise RuntimeError("Could not find train/loss column in CSV")

        for row in reader:
            steps.append(int(row["Step"]))
            losses.append(float(row[loss_key]))

    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.title("Flow Matching Policy Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_TRAIN)
    plt.close()

    print(f"Saved training loss plot to {OUT_TRAIN.resolve()}")


def plot_eval_reward():
    steps = []
    rewards = []

    with open(EVAL_LOG, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            rewards.append(float(row["eval/mean_reward"]))

    plt.figure(figsize=(6, 4))
    plt.plot(steps, rewards)
    plt.xlabel("Training Steps")
    plt.ylabel("Eval Mean Reward")
    plt.title("Flow Matching Policy Evaluation Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_EVAL)
    plt.close()

    print(f"Saved eval reward plot to {OUT_EVAL.resolve()}")


if __name__ == "__main__":
    plot_train_loss()
    plot_eval_reward()
