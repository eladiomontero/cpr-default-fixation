"""
Post-processing pipeline: builds publication-ready LaTeX tables and vector PDF
figures for the "default fixation" CPR-game paper.

Reads (read-only):
  - cprd/output/mult_default/consolidated.csv
  - cprd/output/comprehension_test.csv
Hardcoded human benchmark values (Montero-Porras et al. 2025, Table 1).

Writes:
  - cprd/paper/derived_metrics.csv
  - cprd/paper/tables/design.tex
  - cprd/paper/tables/core_answers.tex
  - cprd/paper/figures/{human_vs_llm,comprehension_vs_pull}.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cprd/
PAPER_DIR = os.path.join(ROOT, "paper")
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CONSOLIDATED_CSV = os.path.join(ROOT, "output", "mult_default", "consolidated.csv")
COMPREHENSION_CSV = os.path.join(ROOT, "output", "comprehension_test.csv")

BASELINE_CONDITION = "no default, no group extraction"
DEFAULTS = [0, 11, 18, 23, 26, 30]

MODELS = [
    "gpt_5.1",
    "gpt_5.4",
    "gpt_5.4_mini",
    "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct",
    "Qwen3_235B_A22B_Instruct",
]

MODEL_META = {
    "gpt_5.1":                  {"provider": "OpenAI",           "params": "undisclosed"},
    "gpt_5.4":                  {"provider": "OpenAI",           "params": "undisclosed"},
    "gpt_5.4_mini":             {"provider": "OpenAI",           "params": "undisclosed"},
    "gpt_5.4_nano":             {"provider": "OpenAI",           "params": "undisclosed"},
    "Llama_3.3_70B_Instruct":   {"provider": "Meta",              "params": "70B"},
    "Qwen3_235B_A22B_Instruct": {"provider": "Alibaba (Qwen team)", "params": "235B (22B active)"},
}

# Fixed colorblind-safe (Okabe-Ito) palette, one color per model, reused
# across Figures 1, 2, and 4.
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
MODEL_COLORS = {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(MODELS)}

# Display names for figures/tables. The underlying model keys above name a
# specific pinned checkpoint/distribution; the paper labels the family name
# only, with the exact distribution specified separately in the text.
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

# Human benchmark (hardcoded; Montero-Porras et al. 2025, Table 1)
HUMAN_BASELINE = 16.32
HUMAN_D11 = 15.72
HUMAN_D23 = 17.84
NASH = 18
SOCIAL_OPTIMUM = 11.5

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
df = pd.read_csv(CONSOLIDATED_CSV)

available_models = set(df["model"].unique())
missing_models = [m for m in MODELS if m not in available_models]
if missing_models:
    print(f"WARNING: models not found in consolidated.csv (no fuzzy match applied, "
          f"check spelling): {missing_models}")

df_sub = df[df["model"].isin(MODELS)].copy()

# ----------------------------------------------------------------------------
# Build derived_metrics.csv
# ----------------------------------------------------------------------------
records = []
baseline_rows = {}
for m in MODELS:
    brow = df_sub[(df_sub["model"] == m) & (df_sub["condition"] == BASELINE_CONDITION)]
    if brow.empty:
        print(f"MISSING: {m} baseline — no valid samples")
        baseline_rows[m] = None
        continue
    baseline_rows[m] = brow.iloc[0]

for m in MODELS:
    brow = baseline_rows[m]
    if brow is None:
        base = np.nan
    else:
        base = brow["average"]

    for d in DEFAULTS:
        arow_df = df_sub[(df_sub["model"] == m) & (df_sub["condition"] == f"default={d}")]
        if arow_df.empty:
            print(f"MISSING: {m} default={d} — no valid samples")
            continue
        arow = arow_df.iloc[0]
        answer = arow["average"]
        if pd.isna(answer):
            print(f"MISSING: {m} default={d} — answer is NaN")
            continue

        # pull
        if pd.isna(base) or abs(d - base) < 1e-9:
            pull = np.nan
        else:
            pull = (answer - base) / (d - base)

        # spike
        pcol = f"p_{d}"
        if pcol not in arow.index or (brow is None) or pcol not in brow.index:
            spike = np.nan
        else:
            p_default_cond = arow[pcol]
            p_baseline_cond = brow[pcol]
            if pd.isna(p_default_cond) or pd.isna(p_baseline_cond):
                spike = np.nan
            else:
                spike = p_default_cond - p_baseline_cond

        records.append({
            "model": m, "default": d, "base": base, "answer": answer,
            "pull": pull, "spike": spike,
        })

derived = pd.DataFrame.from_records(records)
derived_path = os.path.join(PAPER_DIR, "derived_metrics.csv")
derived.to_csv(derived_path, index=False)
print(f"Wrote {derived_path}")

# ----------------------------------------------------------------------------
# Baseline coverage check (sum of p_0..p_30 for baseline rows) — for the
# Qwen footnote in Table 2.
# ----------------------------------------------------------------------------
p_cols = [f"p_{i}" for i in range(31)]
coverage = {}
for m in MODELS:
    brow = baseline_rows[m]
    if brow is None:
        continue
    s = brow[p_cols].astype(float).sum()
    coverage[m] = s
print("Baseline coverage (sum of p_0..p_30) per model:", coverage)

qwen_key = "Qwen3_235B_A22B_Instruct"
qwen_cov = coverage.get(qwen_key, None)
if qwen_cov is not None:
    apply_qwen_footnote = qwen_cov < 0.95
    print(f"Computed Qwen baseline coverage = {qwen_cov:.4f} "
          f"({'<' if apply_qwen_footnote else '>='} 0.95) -> "
          f"{'applying' if apply_qwen_footnote else 'NOT applying'} footnote.")
else:
    apply_qwen_footnote = True
    print("Could not compute Qwen baseline coverage; applying footnote conservatively.")

# ----------------------------------------------------------------------------
# TABLE 1: design.tex
# ----------------------------------------------------------------------------
condition_rows = [
    ("baseline / none", "none"),
    ("default = 0", "abstain"),
    ("default = 11", "social optimum"),
    ("default = 18", "Nash equilibrium"),
    ("default = 23", "exploitative"),
    ("default = 26", "indefensible"),
    ("default = 30", "collapse"),
]

model_lines = []
for m in MODELS:
    meta = MODEL_META[m]
    label = disp_tex(m)
    model_lines.append(f"{label} & {meta['provider']} & {meta['params']} & logprob \\\\")

condition_lines = []
for cond_label, interp in condition_rows:
    condition_lines.append(f"{cond_label} & {interp} \\\\")

design_tex = r"""\begin{table}[t]
\centering
\caption{Experimental design: models and conditions.}
\label{tab:design}

\begin{minipage}{\linewidth}
\centering
\small
\begin{tabular}{llll}
\toprule
Model & Provider & Parameters & Elicitation \\
\midrule
""" + "\n".join(model_lines) + r"""
\bottomrule
\end{tabular}
\end{minipage}

\vspace{1em}

\begin{minipage}{\linewidth}
\centering
\small
\begin{tabular}{ll}
\toprule
Condition & Benchmark interpretation \\
\midrule
""" + "\n".join(condition_lines) + r"""
\bottomrule
\end{tabular}
\end{minipage}

\end{table}
"""

design_path = os.path.join(TABLES_DIR, "design.tex")
with open(design_path, "w") as f:
    f.write(design_tex)
print(f"Wrote {design_path}")

# ----------------------------------------------------------------------------
# TABLE 2: core_answers.tex
# ----------------------------------------------------------------------------
col_defs = ["baseline"] + [f"default={d}" for d in DEFAULTS]

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    base = baseline_rows[m]["average"] if baseline_rows[m] is not None else np.nan
    base_str = f"{base:.1f}" if not pd.isna(base) else "--"
    if m == qwen_key and apply_qwen_footnote:
        base_str += r"$^{\dagger}$"

    vals = []
    for d in DEFAULTS:
        sub = derived[(derived["model"] == m) & (derived["default"] == d)]
        if sub.empty:
            vals.append("--")
        else:
            v = sub.iloc[0]["answer"]
            vals.append(f"{v:.1f}" if not pd.isna(v) else "--")

    rows_tex.append(f"{label} & {base_str} & " + " & ".join(vals) + r" \\")

footnote = ""
if apply_qwen_footnote:
    footnote = (r"\smallskip" + "\n" +
                r"{\footnotesize $^{\dagger}$Reconstructed baseline distribution "
                r"coverage $<0.95$ (conservative caveat).}")

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
# pull@26 (still needed below for the comprehension-vs-pull scatter; the
# pull_toward_26 bar chart and dose_response line plot themselves were
# dropped from the deliverables per the spike-by-anchor rebuild - not
# regenerated here, and any stale copies are deleted by build_spike_by_anchor.py.
# ----------------------------------------------------------------------------
pull26 = derived[derived["default"] == 26][["model", "pull"]].set_index("model")["pull"]
pull26 = pull26.reindex(MODELS)

# ----------------------------------------------------------------------------
# FIGURE 3: human_vs_llm.pdf/.png
# ----------------------------------------------------------------------------
categories = ["baseline", "default=11", "default=23"]
x = np.arange(len(categories))

human_vals = [HUMAN_BASELINE, HUMAN_D11, HUMAN_D23]

llm_base_vals = [baseline_rows[m]["average"] for m in MODELS if baseline_rows[m] is not None]
llm_d11_vals = derived[derived["default"] == 11]["answer"].dropna().tolist()
llm_d23_vals = derived[derived["default"] == 23]["answer"].dropna().tolist()

llm_means = [np.mean(llm_base_vals), np.mean(llm_d11_vals), np.mean(llm_d23_vals)]
llm_mins = [np.min(llm_base_vals), np.min(llm_d11_vals), np.min(llm_d23_vals)]
llm_maxs = [np.max(llm_base_vals), np.max(llm_d11_vals), np.max(llm_d23_vals)]

fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.fill_between(x, llm_mins, llm_maxs, color="#0072B2", alpha=0.2, label="LLM mean (range)")
ax.plot(x, llm_means, marker="o", color="#0072B2", linewidth=1.8, markersize=6, label="LLM mean (range)")
ax.plot(x, human_vals, marker="s", color="black", linewidth=2.4, markersize=6,
        label="Human (Montero-Porras et al. 2025)")

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Extraction")
handles, labels = ax.get_legend_handles_labels()
# dedupe legend (fill_between + plot share label)
seen = {}
for h, l in zip(handles, labels):
    seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "human_vs_llm.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "human_vs_llm.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/human_vs_llm.{pdf,png}")

# ----------------------------------------------------------------------------
# FIGURE 4: comprehension_vs_pull.pdf/.png
# ----------------------------------------------------------------------------
comp = pd.read_csv(COMPREHENSION_CSV)

comp_lookup = {}
comp_models = comp["model"].tolist()
for m in MODELS:
    if m in comp_models:
        comp_lookup[m] = comp.loc[comp["model"] == m, "mean_fraction_correct"].iloc[0]
    else:
        # fuzzy/substring match fallback
        match = [cm for cm in comp_models if m.lower() in cm.lower() or cm.lower() in m.lower()]
        if match:
            comp_lookup[m] = comp.loc[comp["model"] == match[0], "mean_fraction_correct"].iloc[0]
            print(f"NOTE: fuzzy-matched comprehension model '{match[0]}' to consolidated model '{m}'")
        else:
            comp_lookup[m] = np.nan
            print(f"MISSING: {m} — no match found in comprehension_test.csv")

scatter_x, scatter_y, scatter_labels, scatter_colors = [], [], [], []
for m in MODELS:
    cx = comp_lookup.get(m, np.nan)
    cy = pull26.get(m, np.nan)
    if pd.isna(cx) or pd.isna(cy):
        continue
    scatter_x.append(cx)
    scatter_y.append(cy)
    scatter_labels.append(disp(m))
    scatter_colors.append(MODEL_COLORS[m])

# correlation check (comprehension vs pull@26 only, drives the trend line
# in the figure below)
r, p = (np.nan, np.nan)
if len(scatter_x) >= 3:
    r, p = stats.pearsonr(scatter_x, scatter_y)
print(f"Pearson correlation comprehension vs pull@26: r={r:.3f}, p={p:.3f} "
      f"({'significant' if (not pd.isna(p) and p < 0.05) else 'NOT significant'} at alpha=0.05) "
      f"-> {'drawing' if (not pd.isna(p) and p < 0.05) else 'NOT drawing'} trend line.")

# pooled correlation: comprehension vs pull across ALL defaults (each
# model's comprehension score repeated against its pull at every default,
# i.e. up to 6 models x 6 defaults = 36 points), not just default=26.
pooled_x, pooled_y = [], []
for m in MODELS:
    cx = comp_lookup.get(m, np.nan)
    if pd.isna(cx):
        continue
    sub = derived[derived["model"] == m].dropna(subset=["pull"])
    for _, row in sub.iterrows():
        pooled_x.append(cx)
        pooled_y.append(row["pull"])

r_pooled, p_pooled = (np.nan, np.nan)
if len(pooled_x) >= 3:
    r_pooled, p_pooled = stats.pearsonr(pooled_x, pooled_y)
print(f"Pearson correlation comprehension vs pull, POOLED across all {len(DEFAULTS)} defaults "
      f"(n={len(pooled_x)} model-default pairs): r={r_pooled:.3f}, p={p_pooled:.3f} "
      f"({'significant' if (not pd.isna(p_pooled) and p_pooled < 0.05) else 'NOT significant'} at alpha=0.05). "
      f"Note: comprehension is repeated across each model's 6 defaults, so these points are "
      f"not independent - read this as descriptive, not a clean hypothesis test.")

fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.scatter(scatter_x, scatter_y, c=scatter_colors, edgecolor="black", linewidth=0.5, s=50, zorder=3)
for xx, yy, lab in zip(scatter_x, scatter_y, scatter_labels):
    ax.annotate(lab, (xx, yy), textcoords="offset points", xytext=(6, 4), fontsize=8)

if not pd.isna(p) and p < 0.05:
    coeffs = np.polyfit(scatter_x, scatter_y, 1)
    xs_line = np.linspace(min(scatter_x), max(scatter_x), 50)
    ax.plot(xs_line, np.polyval(coeffs, xs_line), linestyle="--", color="gray", linewidth=1)

ax.set_xlim(left=0)
ax.set_xlabel("Comprehension (mean fraction correct)")
ax.set_ylabel("Pull toward 26")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "comprehension_vs_pull.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "comprehension_vs_pull.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/comprehension_vs_pull.{pdf,png}")

# ----------------------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------------------
print("\n=== FILES WRITTEN ===")
for p in [
    derived_path, design_path, core_path,
    os.path.join(FIGURES_DIR, "human_vs_llm.pdf"),
    os.path.join(FIGURES_DIR, "human_vs_llm.png"),
    os.path.join(FIGURES_DIR, "comprehension_vs_pull.pdf"),
    os.path.join(FIGURES_DIR, "comprehension_vs_pull.png"),
]:
    print(p)

print("\n=== SANITY SUMMARY (baseline, answer@26, pull@26) ===")
for m in MODELS:
    base = baseline_rows[m]["average"] if baseline_rows[m] is not None else np.nan
    sub = derived[(derived["model"] == m) & (derived["default"] == 26)]
    ans = sub.iloc[0]["answer"] if not sub.empty else np.nan
    pull = sub.iloc[0]["pull"] if not sub.empty else np.nan
    print(f"{m}: baseline={base:.2f}, answer@26={ans:.2f}, pull@26={pull:.3f}" if not pd.isna(ans)
          else f"{m}: baseline={base}, answer@26=MISSING, pull@26=MISSING")
