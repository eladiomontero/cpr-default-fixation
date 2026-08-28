"""Rebuild the CPR paper's headline figures/tables around spike, framed
column-first by anchor (default value) rather than by model. Drops pull and
the dose-response line plot per the latest brief.

Reads (read-only): cprd/paper/derived_metrics.csv
Writes:
  - cprd/paper/figures/{spike_heatmap,spike_at_26}.{pdf,png}
  - cprd/paper/tables/{core_answers,spike_table}.tex
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
DERIVED_CSV = os.path.join(PAPER_DIR, "derived_metrics.csv")

MODELS = [
    "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
]
DEFAULTS = [0, 11, 18, 23, 26, 30]
BENCHMARK_LABEL = {
    0: "abstain", 11: "social opt", 18: "Nash",
    23: "exploit", 26: "indefensible", 30: "collapse",
}
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


def disp_tex(m: str) -> str:
    return disp(m).replace("_", r"\_")


plt.rcParams.update({"font.size": 11, "font.family": "serif"})

derived = pd.read_csv(DERIVED_CSV)
derived_sub = derived[derived["model"].isin(MODELS)].copy()

# ----------------------------------------------------------------------------
# SANITY GATE
# ----------------------------------------------------------------------------
qwen_base_rows = derived_sub.loc[derived_sub["model"] == "Qwen3_235B_A22B_Instruct", "base"]
if qwen_base_rows.empty or pd.isna(qwen_base_rows.iloc[0]):
    print("MISSING: Qwen3_235B_A22B_Instruct base — cannot run sanity gate. Aborting.")
    sys.exit(1)
qwen_base = qwen_base_rows.iloc[0]
print(f"Qwen3_235B_A22B_Instruct baseline = {qwen_base:.3f}")
if qwen_base < 5:
    print("STALE QWEN — baseline < 5. Aborting, no figures/tables written.")
    sys.exit(1)
print("Sanity gate PASSED (baseline >= 5).")

# ----------------------------------------------------------------------------
# Baseline / answer / spike lookups
# ----------------------------------------------------------------------------
base_by_model = {}
for m in MODELS:
    rows = derived_sub.loc[derived_sub["model"] == m, "base"]
    if rows.empty or pd.isna(rows.iloc[0]):
        print(f"MISSING: {m} base")
        base_by_model[m] = np.nan
    else:
        base_by_model[m] = rows.iloc[0]

spike_pivot = derived_sub.pivot_table(index="model", columns="default", values="spike", aggfunc="first")
spike_pivot = spike_pivot.reindex(index=MODELS, columns=DEFAULTS)
for m in MODELS:
    for d in DEFAULTS:
        if pd.isna(spike_pivot.loc[m, d]):
            print(f"MISSING: {m} spike@{d}")

answer_pivot = derived_sub.pivot_table(index="model", columns="default", values="answer", aggfunc="first")
answer_pivot = answer_pivot.reindex(index=MODELS, columns=DEFAULTS)

# row order: mean spike across all defaults, descending
mean_spike_per_model = spike_pivot.mean(axis=1, skipna=True)
row_order = mean_spike_per_model.reindex(MODELS).sort_values(ascending=False, na_position="last").index.tolist()

# column mean (bottom summary row): mean spike across the 6 models per default
col_mean = spike_pivot.mean(axis=0, skipna=True).reindex(DEFAULTS)

# ----------------------------------------------------------------------------
# FIGURE 1: spike_heatmap.pdf (column-first, with bottom "column mean" row)
# ----------------------------------------------------------------------------
heat = spike_pivot.reindex(row_order).to_numpy(dtype=float)
full = np.vstack([heat, col_mean.to_numpy(dtype=float)[None, :]])
row_labels = [disp(m) for m in row_order] + ["column mean"]

fig, ax = plt.subplots(figsize=(5.0, 3.6))
im = ax.imshow(full, cmap="Blues", vmin=0, vmax=1, aspect="auto")

n_model_rows = len(row_order)
ax.axhline(n_model_rows - 0.5, color="black", linewidth=1.2)

ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
for lbl in ax.get_yticklabels()[n_model_rows:]:
    lbl.set_fontweight("bold")

ax.set_xticks(range(len(DEFAULTS)))
xticklabels = [f"{d}\n{BENCHMARK_LABEL[d]}" for d in DEFAULTS]
ax.set_xticklabels(xticklabels, fontsize=7)

for i in range(full.shape[0]):
    for j in range(full.shape[1]):
        v = full[i, j]
        if np.isnan(v):
            ax.text(j, i, "NaN", ha="center", va="center", fontsize=7, color="black")
            continue
        color = "white" if v > 0.55 else "black"
        weight = "bold" if i == n_model_rows else "normal"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=color, fontweight=weight)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Prob. mass on exact default", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "spike_heatmap.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "spike_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/spike_heatmap.{pdf,png}")

# ----------------------------------------------------------------------------
# FIGURE 2: spike_at_26.pdf
# ----------------------------------------------------------------------------
spike26 = spike_pivot[26].reindex(MODELS)
sorted26 = spike26.sort_values(ascending=False, na_position="last").index.tolist()

fig, ax = plt.subplots(figsize=(3.4, 3.2))
xs = np.arange(len(sorted26))
vals = [spike26[m] for m in sorted26]
plot_vals = [0 if pd.isna(v) else v for v in vals]
colors = [MODEL_COLORS[m] for m in sorted26]
ax.bar(xs, plot_vals, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(vals):
    label = "NaN" if pd.isna(v) else f"{v:.2f}"
    y = 0.02 if pd.isna(v) else v + 0.02
    ax.text(i, y, label, ha="center", va="bottom", fontsize=8)

ax.set_xticks(xs)
ax.set_xticklabels([disp(m) for m in sorted26], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Prob. mass on default = 26")
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "spike_at_26.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "spike_at_26.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/spike_at_26.{pdf,png}")

# ----------------------------------------------------------------------------
# TABLE 2: core_answers.tex (kept, corrected Qwen; 1 decimal)
# ----------------------------------------------------------------------------
p_cols_needed = ["p_0", "p_1"]  # not used here; coverage handled via derived's base/answer only
qwen_key = "Qwen3_235B_A22B_Instruct"

# Coverage footnote: derived_metrics.csv has no p_ columns, so fall back to the
# known caveat already established for Qwen's baseline reconstruction (coverage
# 0.884, computed previously from consolidated.csv) rather than recomputing here
# (out of scope: this script only reads derived_metrics.csv per the brief).
QWEN_BASELINE_COVERAGE = 0.8844
apply_qwen_footnote = QWEN_BASELINE_COVERAGE < 0.95

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    base = base_by_model[m]
    base_str = f"{base:.1f}" if not pd.isna(base) else "--"
    if m == qwen_key and apply_qwen_footnote:
        base_str += r"$^{\dagger}$"
    vals = []
    for d in DEFAULTS:
        v = answer_pivot.loc[m, d]
        vals.append(f"{v:.1f}" if not pd.isna(v) else "--")
    rows_tex.append(f"{label} & {base_str} & " + " & ".join(vals) + r" \\")

footnote = ""
if apply_qwen_footnote:
    footnote = (r"\smallskip" + "\n" +
                r"{\footnotesize $^{\dagger}$Reconstructed baseline distribution "
                rf"coverage $={QWEN_BASELINE_COVERAGE:.3f} < 0.95$.}}")

core_tex = r"""\begin{table}[t]
\centering
\caption{Model answers by condition (mean extraction, integer scale 0--30).}
\label{tab:core_answers}
\small
\begin{tabular}{lrrrrrrr}
\toprule
Model & Baseline & $d{=}0$ & $d{=}11$ & $d{=}18$ & $d{=}23$ & $d{=}26$ & $d{=}30$ \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}

""" + footnote + r"""
\end{table}
"""
core_path = os.path.join(TABLES_DIR, "core_answers.tex")
with open(core_path, "w") as f:
    f.write(core_tex)
print(f"Wrote {core_path}")

# ----------------------------------------------------------------------------
# TABLE 3: spike_table.tex (NEW)
# ----------------------------------------------------------------------------
rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    vals = []
    for d in DEFAULTS:
        v = spike_pivot.loc[m, d]
        vals.append(f"{v:.2f}" if not pd.isna(v) else "--")
    mean_v = mean_spike_per_model[m]
    mean_str = f"{mean_v:.2f}" if not pd.isna(mean_v) else "--"
    rows_tex.append(f"{label} & " + " & ".join(vals) + f" & {mean_str}" + r" \\")

col_mean_row = "column mean & " + " & ".join(
    f"{col_mean[d]:.2f}" if not pd.isna(col_mean[d]) else "--" for d in DEFAULTS
) + f" & {col_mean.mean():.2f}" + r" \\"

spike_tex = r"""\begin{table}[t]
\centering
\caption{Spike: probability mass placed on the exact default value (0--1), by model and default. Rows sorted by mean spike, descending.}
\label{tab:spike_table}
\small
\begin{tabular}{lrrrrrrr}
\toprule
Model & $d{=}0$ & $d{=}11$ & $d{=}18$ & $d{=}23$ & $d{=}26$ & $d{=}30$ & Mean \\
\midrule
""" + "\n".join(f"{disp_tex(m)} & " + " & ".join(
    f"{spike_pivot.loc[m, d]:.2f}" if not pd.isna(spike_pivot.loc[m, d]) else "--" for d in DEFAULTS
) + f" & {mean_spike_per_model[m]:.2f}" + r" \\" for m in row_order) + r"""
\midrule
""" + col_mean_row + r"""
\bottomrule
\end{tabular}
\end{table}
"""
spike_table_path = os.path.join(TABLES_DIR, "spike_table.tex")
with open(spike_table_path, "w") as f:
    f.write(spike_tex)
print(f"Wrote {spike_table_path}")

# ----------------------------------------------------------------------------
# SANITY SUMMARY
# ----------------------------------------------------------------------------
print("\n=== FILES WRITTEN ===")
for p in [
    os.path.join(FIGURES_DIR, "spike_heatmap.pdf"),
    os.path.join(FIGURES_DIR, "spike_heatmap.png"),
    os.path.join(FIGURES_DIR, "spike_at_26.pdf"),
    os.path.join(FIGURES_DIR, "spike_at_26.png"),
    core_path,
    spike_table_path,
]:
    print(p)

print(f"\nQwen baseline: {qwen_base:.3f} (>= 5: {'YES' if qwen_base >= 5 else 'NO'})")
print("\nColumn-mean spike per default:")
for d in DEFAULTS:
    print(f"  default={d} ({BENCHMARK_LABEL[d]}): {col_mean[d]:.3f}")
print("\nSpike@26 per model:")
for m in sorted26:
    v = spike26[m]
    print(f"  {m}: {v:.3f}" if not pd.isna(v) else f"  {m}: MISSING")
