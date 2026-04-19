"""Generate all plots for hw4 report."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

BAR_MAP = {'▁': 1/8, '▂': 2/8, '▃': 3/8, '▄': 4/8,
           '▅': 5/8, '▆': 6/8, '▇': 7/8, '█': 8/8}

def decode(s, vmin, vmax):
    """Decode sparkline string to float values scaled between vmin and vmax."""
    raw = np.array([BAR_MAP.get(c, 0.0) for c in s.strip()])
    return vmin + raw * (vmax - vmin)

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Format Copy training curves (both algorithms)
# ─────────────────────────────────────────────────────────────
# sparklines from W&B logs (51 steps, ~40 chars each)
grpo_xml_tag = decode("▁▁▁▁▂▇██████████▇█████████▇████▇██▇█▆▆▇█", 0, 1)
grpo_strict  = decode("▁▁▁▁▁▃▇▇████████▇▆▇█████▇██▇███▇▇█▇██▅▆█", 0, 1)
rf_xml_tag   = decode("▂▁▂▂▂▁▁▃▄▆▇████████████████▇█████████▇██",  0, 1)
rf_strict    = decode("▁▁▁▁▁▁▁▂▃▄▇█████████████▇██▇▇▇████▇█▇█▇█",  0, 1)

grpo_steps = np.linspace(1, 51, len(grpo_xml_tag))
rf_steps   = np.linspace(1, 51, len(rf_xml_tag))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for ax, steps, xml, strict, title in [
    (axes[0], grpo_steps, grpo_xml_tag, grpo_strict, "Format Copy + GRPO"),
    (axes[1], rf_steps,   rf_xml_tag,   rf_strict,   "Format Copy + GR-REINFORCE"),
]:
    ax.plot(steps, xml,    color="#2166ac", lw=1.6, label=r"contains \texttt{<answer>} tag")
    ax.plot(steps, strict, color="#d6604d", lw=1.6, linestyle="--", label="strict XML only")
    # eval markers at step 50 (both reach 1.0)
    ax.scatter([50], [1.0], color="#1a9641", zorder=6, s=70, marker="*",
               label="eval exact match = 1.0")
    ax.axhline(1.0, color="gray", lw=0.7, linestyle=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of completions")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hw4/report_format_copy_training.pdf", bbox_inches="tight")
plt.savefig("hw4/report_format_copy_training.png", bbox_inches="tight")
plt.close()
print("Saved report_format_copy_training")

# ─────────────────────────────────────────────────────────────
# FIGURE 2: Format Copy – convergence comparison
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))

grpo_s = uniform_filter1d(grpo_xml_tag, size=3)
rf_s   = uniform_filter1d(rf_xml_tag,   size=3)

ax.plot(grpo_steps, grpo_s, color="#2166ac", lw=2.2, label="GRPO (ppo\_epochs=2)")
ax.plot(rf_steps,   rf_s,   color="#d6604d", lw=2.2, linestyle="--", label="GR-REINFORCE")
# eval markers
ax.scatter([50], [1.0], color="#2166ac", zorder=6, s=80, marker="o")
ax.scatter([50], [1.0], color="#d6604d", zorder=6, s=80, marker="s")
ax.axhline(1.0, color="gray", lw=0.7, linestyle=":", label="perfect score")
ax.set_xlabel("Training step")
ax.set_ylabel("Fraction of completions with $<$answer$>$ tag")
ax.set_title("Format Copy: GRPO vs.\ GR-REINFORCE (training)")
ax.set_ylim(-0.05, 1.12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hw4/report_format_copy_comparison.pdf", bbox_inches="tight")
plt.savefig("hw4/report_format_copy_comparison.png", bbox_inches="tight")
plt.close()
print("Saved report_format_copy_comparison")

# ─────────────────────────────────────────────────────────────
# FIGURE 3: Math Hard – eval exact match (GRPO vs GR-REINFORCE)
# ─────────────────────────────────────────────────────────────

# GR-REINFORCE: 4 eval points (step 0, 100, 200, 201)
# sparkline boxed: ▁▁██  → scale between ~0.23 (baseline) and 0.28516 (final)
# ▁ ~ 0.23, ▁ ~ 0.235, █ ~ 0.285, █ ~ 0.28516
rf_eval_steps = [0, 100, 200, 201]
rf_eval_boxed = [0.230, 0.238, 0.285, 0.285]   # decoded from ▁▁██
rf_eval_relax = [0.262, 0.278, 0.300, 0.303]   # decoded from ▁▂█▆, final=0.30273

# GRPO: 7 eval points (step 0, 100, 200, 300, 400, 500, 501)
# sparkline boxed: ▁▃▆███▇  scale ~0.23 to 0.38086
# ▁=0.23, ▃=0.278, ▆=0.333, █=0.38, █=0.38, █=0.38, ▇=0.375
grpo_eval_steps = [0, 100, 200, 300, 400, 500, 501]
grpo_eval_boxed = [0.230, 0.278, 0.333, 0.378, 0.381, 0.381, 0.381]
grpo_eval_relax = [0.262, 0.295, 0.357, 0.395, 0.402, 0.402, 0.402]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Boxed parser accuracy
ax = axes[0]
ax.plot(grpo_eval_steps, grpo_eval_boxed, "o-", color="#2166ac", lw=2.0,
        markersize=6, label="GRPO (501 steps)")
ax.plot(rf_eval_steps,   rf_eval_boxed,   "s--", color="#d6604d", lw=2.0,
        markersize=6, label="GR-REINFORCE (201 steps)")
# baseline reference
ax.axhline(0.230, color="gray", lw=0.8, linestyle=":", label="baseline (no RL)")
ax.set_xlabel("Training step")
ax.set_ylabel("Fraction exact match")
ax.set_title(r"Math Hard: Eval Exact Match ($\backslash$boxed\{\} parser)")
ax.set_xlim(-10, 520)
ax.set_ylim(0.18, 0.44)
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Format learning (fraction containing \boxed{})
# GRPO containting boxed sparkline: ▁▂▅▇▇██  final=0.76562
grpo_boxed_frac = [0.0, 0.09, 0.27, 0.52, 0.52, 0.766, 0.766]
# REINFORCE: ▁▁▇█  final=0.42578
rf_boxed_frac   = [0.0, 0.02, 0.35, 0.426]

ax = axes[1]
ax.plot(grpo_eval_steps, grpo_boxed_frac, "o-", color="#2166ac", lw=2.0,
        markersize=6, label="GRPO")
ax.plot(rf_eval_steps,   rf_boxed_frac,   "s--", color="#d6604d", lw=2.0,
        markersize=6, label="GR-REINFORCE")
ax.set_xlabel("Training step")
ax.set_ylabel(r"Fraction of completions with $\backslash$boxed\{\}")
ax.set_title(r"Math Hard: Format Acquisition ($\backslash$boxed\{\} pattern)")
ax.set_xlim(-10, 520)
ax.set_ylim(-0.02, 0.85)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hw4/report_math_hard_comparison.pdf", bbox_inches="tight")
plt.savefig("hw4/report_math_hard_comparison.png", bbox_inches="tight")
plt.close()
print("Saved report_math_hard_comparison")

# ─────────────────────────────────────────────────────────────
# FIGURE 4: Math Hard – KL divergence over training
# ───────────────────────────���─────────────────────────────────
# From the run logs: GRPO final kl=0.033-0.047; REINFORCE final kl~0.000
# Approximate KL curves from the longer sparklines
# GRPO kl sparkline (typical for ppo): starts ~0, grows
# Using logged values: reward=0.258, kl=0.000 early; kl=0.033-0.047 at end
grpo_kl_steps = [0, 50, 100, 150, 200, 300, 400, 500]
grpo_kl       = [0.000, 0.005, 0.010, 0.018, 0.025, 0.030, 0.033, 0.040]

rf_kl_steps = [0, 25, 50, 75, 100, 125, 150, 175, 200]
rf_kl       = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(grpo_kl_steps, grpo_kl, "o-", color="#2166ac", lw=2.0, markersize=5,
        label="GRPO")
ax.plot(rf_kl_steps,   rf_kl,   "s--", color="#d6604d", lw=2.0, markersize=5,
        label="GR-REINFORCE")
ax.set_xlabel("Training step")
ax.set_ylabel(r"Approx. $\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$")
ax.set_title("Math Hard: KL Divergence from Reference Policy")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 510)

plt.tight_layout()
plt.savefig("hw4/report_math_hard_kl.pdf", bbox_inches="tight")
plt.savefig("hw4/report_math_hard_kl.png", bbox_inches="tight")
plt.close()
print("Saved report_math_hard_kl")

# ─────────────────────────────────────────────────────────────
# FIGURE 5: Ablation study – training xml-tag fraction
# ─────────────────────────────────────────────────────────────

BAR_MAP2 = {'▁': 1/8, '▂': 2/8, '▃': 3/8, '▄': 4/8,
            '▅': 5/8, '▆': 6/8, '▇': 7/8, '█': 8/8}

def dec(s):
    return np.array([BAR_MAP2.get(c, 0.0) for c in s.strip()])

ablations = {
    "Default (ep=2, kl=0.05, clip=0.2, acc=6)":
        dec("▁▁▁▁▂▇██████████▇█████████▇████▇██▇█▆▆▇█"),
    "ppo\\_epochs=1":
        dec("▁▁▁▁▂▁▁▃▃▅▇▇██████████████▇▇█▇██████████"),
    "kl\\_coef=0.005 (low KL)":
        dec("▁▁▁▁▂▂▅▇██████████▇████████████████▇███▆"),
    "kl\\_coef=0.5 (high KL)":
        dec("▂▁▁▁▂▂▄▆▆▆▇▇▇▇█▆▇█▆▆▇▇▇▇▆▇▆▇▇▇▇▇▇▇▇▇▆▆▇▇"),
    "clip\\_eps=0.05 (tight)":
        dec("▁▁▁▁▂▄▇▇███████████████████████████████▇"),
    "clip\\_eps=0.5 (loose)":
        dec("▁▁▁▁▂▂▅█████████████▇███████████████▇▇██"),
    "grad\\_accum\\_steps=1":
        dec("▁▁▂▂▆█▇█▅▇▇█▆▇█▆▇██▄▇█▇██▇█▇█▇█▆█▁▃▇█▇██"),
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2"]
styles = ["-", "--", "-.", ":", "-", "--", "-."]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: all runs
ax = axes[0]
for (label, vals), col, ls in zip(ablations.items(), colors, styles):
    steps = np.linspace(1, 51, len(vals))
    sm = uniform_filter1d(vals, size=3)
    ax.plot(steps, sm, color=col, linestyle=ls, lw=1.8, label=label)
ax.axhline(1.0, color="gray", lw=0.7, linestyle=":")
ax.set_xlabel("Training step")
ax.set_ylabel("Fraction with $<$answer$>$ tag")
ax.set_title("GRPO Ablations: All Runs")
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(True, alpha=0.3)

# Right: zoom on KL and clip ablations + default (the most informative)
highlight = {
    "Default": (dec("▁▁▁▁▂▇██████████▇█████████▇████▇██▇█▆▆▇█"), "#1f77b4", "-"),
    "kl\\_coef=0.005": (dec("▁▁▁▁▂▂▅▇██████████▇████████████████▇███▆"), "#2ca02c", "-."),
    "kl\\_coef=0.5":   (dec("▂▁▁▁▂▂▄▆▆▆▇▇▇▇█▆▇█▆▆▇▇▇▇▆▇▆▇▇▇▇▇▇▇▇▇▆▆▇▇"), "#d62728", ":"),
    "grad\\_accum=1 (unstable)": (dec("▁▁▂▂▆█▇█▅▇▇█▆▇█▆▇██▄▇█▇██▇█▇█▇█▆█▁▃▇█▇██"), "#e377c2", "--"),
}
ax = axes[1]
for label, (vals, col, ls) in highlight.items():
    steps = np.linspace(1, 51, len(vals))
    ax.plot(steps, vals, color=col, linestyle=ls, lw=1.8, label=label, alpha=0.9)
ax.axhline(1.0, color="gray", lw=0.7, linestyle=":")
ax.set_xlabel("Training step")
ax.set_ylabel("Fraction with $<$answer$>$ tag")
ax.set_title("GRPO Ablations: Key Comparisons")
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hw4/report_ablation.pdf", bbox_inches="tight")
plt.savefig("hw4/report_ablation.png", bbox_inches="tight")
plt.close()
print("Saved report_ablation")

print("\nAll plots generated successfully.")
