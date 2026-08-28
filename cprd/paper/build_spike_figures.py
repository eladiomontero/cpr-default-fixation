"""Two additional figures for the corrected-Qwen build: spike_heatmap.pdf
(headline) and distributions_at_26.pdf (mechanistic punchline). Reuses the
same MODELS / MODEL_COLORS / derived_metrics.csv produced by
build_paper_assets.py, which must be run first.

Reads (read-only):
  - cprd/output/mult_default/consolidated.csv
  - cprd/paper/derived_metrics.csv
Writes:
  - cprd/paper/figures/{spike_heatmap,distributions_at_26}.{pdf,png}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
CONSOLIDATED_CSV = os.path.join(ROOT, "output", "mult_default", "consolidated.csv")
DERIVED_CSV = os.path.join(PAPER_DIR, "derived_metrics.csv")

BASELINE_CONDITION = "no default, no group extraction"
DEFAULTS = [0, 11, 18, 23, 26, 30]
MODELS = [
    "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
]
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
MODEL_COLORS = {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(MODELS)}

DISPLAY_NAMES = {
    "gpt_5.1": "gpt_5.1",
    "gpt_5.4": "gpt_5.4",
    "gpt_5.4_mini": "gpt_5.4_mini",
    "gpt_5.4_nano": "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct": "Llama3.3",
    "Qwen3_235B_A22B_Instruct": "Qwen3",
}


def disp(m: str) -> str:
    return DISPLAY_NAMES.get(m, m.replace("_", " "))


plt.rcParams.update({"font.size": 11, "font.family": "serif"})

df = pd.read_csv(CONSOLIDATED_CSV)
df_sub = df[df["model"].isin(MODELS)].copy()
derived = pd.read_csv(DERIVED_CSV)

# ----------------------------------------------------------------------------
# FIGURE 1: spike_heatmap.pdf
# ----------------------------------------------------------------------------
spike_pivot = derived.pivot(index="model", columns="default", values="spike")
spike_pivot = spike_pivot.reindex(columns=DEFAULTS)

mean_spike = spike_pivot.mean(axis=1, skipna=True)
row_order = mean_spike.sort_values(ascending=False, na_position="last").index.tolist()
spike_pivot = spike_pivot.reindex(row_order)

fig, ax = plt.subplots(figsize=(3.4, 3.4))
data = spike_pivot.to_numpy(dtype=float)
im = ax.imshow(data, cmap="Greys", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(DEFAULTS)))
ax.set_xticklabels(DEFAULTS)
ax.set_yticks(range(len(row_order)))
ax.set_yticklabels([disp(m) for m in row_order], fontsize=8)
ax.set_xlabel("Default value")

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if np.isnan(v):
            ax.text(j, i, "NaN", ha="center", va="center", fontsize=7, color="black")
            continue
        color = "white" if v > 0.55 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Prob. mass on exact default", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "spike_heatmap.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "spike_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/spike_heatmap.{pdf,png}")
print("Row order (mean spike descending):", row_order)
print("Mean spike per model:")
for m in row_order:
    print(f"  {m}: {mean_spike[m]:.3f}")

# ----------------------------------------------------------------------------
# FIGURE 2: distributions_at_26.pdf (small multiples, default=26 only)
# ----------------------------------------------------------------------------
p_cols_order = [f"p_{i}" for i in range(31)]

fig, axes = plt.subplots(2, 3, figsize=(6.0, 4.0), sharey=True)
axes_flat = axes.flatten()

for ax, m in zip(axes_flat, MODELS):
    row = df_sub[(df_sub["model"] == m) & (df_sub["condition"] == "default=26")]
    if row.empty:
        ax.set_title(disp(m), fontsize=9)
        ax.text(0.5, 0.5, "MISSING", ha="center", va="center", transform=ax.transAxes)
        print(f"MISSING: {m} default=26 distribution — skipping panel data")
        continue
    row = row.iloc[0]
    vals = row[p_cols_order].astype(float).fillna(0.0).to_numpy()
    xs = np.arange(31)
    ax.bar(xs, vals, color=MODEL_COLORS[m], width=0.9)
    ax.axvline(26, color="black", linestyle="--", linewidth=1)
    ax.set_title(disp(m), fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xticks([0, 10, 20, 26, 30])

for ax in axes_flat[3:]:
    ax.set_xlabel("Answer", fontsize=8)
for ax in axes_flat[::3]:
    ax.set_ylabel("Probability", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "distributions_at_26.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "distributions_at_26.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/distributions_at_26.{pdf,png}")
