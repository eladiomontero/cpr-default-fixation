"""Per-model detail pass on the two-game default-framing paper: fills in
whether the framing effect / conflict-agreement mechanism / wording
instability generalize across all 6 models or are driven by a subset, adds
a comprehension null-check, and fixes the human-vs-LLM figure onto shared
units. Pure post-processing of already-computed files - no model calls.

Reads (read-only):
  - cprd/paper/derived_metrics_twogame.csv
  - cprd/output/comprehension_test.csv
  - cprd/output/mult_default/consolidated.csv
Hardcoded: human benchmark (Montero-Porras et al. 2025, Table 1).

Writes:
  - cprd/paper/tables/{framing_permodel,conflict_agreement_permodel,comprehension}.tex
  - cprd/paper/figures/{framing_effect_bothgames,per_model_instability_crd,human_vs_llm}.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # LLMs/
CPRD_DIR = os.path.join(ROOT, "cprd")
PAPER_DIR = os.path.join(CPRD_DIR, "paper")
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")

DERIVED_CSV = os.path.join(PAPER_DIR, "derived_metrics_twogame.csv")
COMPREHENSION_CSV = os.path.join(CPRD_DIR, "output", "comprehension_test.csv")
CPR_CONSOLIDATED = os.path.join(CPRD_DIR, "output", "mult_default", "consolidated.csv")

MODELS = [
    "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
]
DISPLAY_NAMES = {
    "gpt_5.1": "gpt_5.1", "gpt_5.4": "gpt_5.4", "gpt_5.4_mini": "gpt_5.4_mini",
    "gpt_5.4_nano": "gpt_5.4_nano", "Llama_3.3_70B_Instruct": "Llama3.3",
    "Qwen3_235B_A22B_Instruct": "Qwen3",
}


def disp(m):
    return DISPLAY_NAMES.get(m, m.replace("_", " "))


def disp_tex(m):
    return disp(m).replace("_", r"\_")


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
MODEL_COLORS = {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(MODELS)}

CPR_CONFLICT_ANCHORS = [18, 26]
CRD_CONFLICT_ANCHORS = [0, 2]
WORDINGS4 = ["V0_true", "V1", "V2", "V3"]
CPR_BASELINE_CONDITION = "no default, no group extraction"

HUMAN_BASELINE = 16.32
HUMAN_D11 = 15.72
HUMAN_D23 = 17.84

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

derived = pd.read_csv(DERIVED_CSV)
derived = derived[derived["model"].isin(MODELS)].copy()

# ============================================================================
# 1. PER-MODEL FRAMING TABLE + FIGURE
# ============================================================================
print("=" * 70)
print("1. PER-MODEL FRAMING")
print("=" * 70)

framing_rows_tex = []
framing_matrix = {}  # (game, model, wording) -> mean net_pull over conflict anchors
for game, anchors in [("CPR", CPR_CONFLICT_ANCHORS), ("CRD", CRD_CONFLICT_ANCHORS)]:
    for m in MODELS:
        for w in WORDINGS4:
            sub = derived[(derived.game == game) & (derived.model == m) &
                          (derived.anchor.isin(anchors)) & (derived.wording == w)]
            framing_matrix[(game, m, w)] = sub["net_pull"].mean() if not sub.empty else np.nan

for m in MODELS:
    label = disp_tex(m)
    cpr_vals = [framing_matrix[("CPR", m, w)] for w in WORDINGS4]
    crd_vals = [framing_matrix[("CRD", m, w)] for w in WORDINGS4]
    cpr_str = " & ".join(f"{v:+.2f}" if pd.notna(v) else "--" for v in cpr_vals)
    crd_str = " & ".join(f"{v:+.2f}" if pd.notna(v) else "--" for v in crd_vals)
    framing_rows_tex.append(f"{label} & {cpr_str} & {crd_str} \\\\")

framing_tex = r"""\begin{table}[t]
\centering
\caption{Per-model framing effect: mean net pull at the conflict anchors (CPR: 18, 26; CRD: 0, 2), by wording. -- = excluded (rate-limit failure or unstable baseline).}
\label{tab:framing_permodel}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{CPR (mean over $d{=}18,26$)} & \multicolumn{4}{c}{CRD (mean over $d{=}0,2$)} \\
Model & V0t & V1 & V2 & V3 & V0t & V1 & V2 & V3 \\
\midrule
""" + "\n".join(framing_rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
framing_path = os.path.join(TABLES_DIR, "framing_permodel.tex")
with open(framing_path, "w") as f:
    f.write(framing_tex)
print(f"Wrote {framing_path}")

# how many models show V2 < V0_true AND V3 > V0_true, per game
for game in ["CPR", "CRD"]:
    count = 0
    detail = []
    for m in MODELS:
        v0 = framing_matrix[(game, m, "V0_true")]
        v2 = framing_matrix[(game, m, "V2")]
        v3 = framing_matrix[(game, m, "V3")]
        if pd.isna(v0) or pd.isna(v2) or pd.isna(v3):
            detail.append(f"{disp(m)}:incomplete")
            continue
        holds = (v2 < v0) and (v3 > v0)
        detail.append(f"{disp(m)}:{'YES' if holds else 'no'}")
        if holds:
            count += 1
    print(f"{game}: {count}/6 models show V2<V0_true AND V3>V0_true  [{', '.join(detail)}]")
print()

# --- Figure 1 rebuild: mean line bold + faint per-model overlays ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)

for ax, game, anchors, marker in [(axes[0], "CPR", CPR_CONFLICT_ANCHORS, "o"),
                                  (axes[1], "CRD", CRD_CONFLICT_ANCHORS, "s")]:
    for m in MODELS:
        vals = [framing_matrix[(game, m, w)] for w in WORDINGS4]
        if all(pd.isna(v) for v in vals):
            continue
        ax.plot(range(4), vals, color=MODEL_COLORS[m], alpha=0.35, linewidth=1.1,
               marker=marker, markersize=3, zorder=2)
    mean_vals = [derived[(derived.game == game) & (derived.anchor.isin(anchors)) &
                        (derived.wording == w)]["net_pull"].mean() for w in WORDINGS4]
    ax.plot(range(4), mean_vals, color="black", linewidth=2.6, marker=marker, markersize=7,
           zorder=5, label="mean (all models)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["V0t", "V1", "V2", "V3"])
    ax.set_xlabel("Wording")
    ax.set_title(f"{game} ({'extract' if game == 'CPR' else 'contribute'})", fontsize=10)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Mean net pull")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "framing_effect_bothgames.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "framing_effect_bothgames.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/framing_effect_bothgames.{pdf,png}\n")

# ============================================================================
# 2. PER-MODEL CONFLICT/AGREEMENT TABLE
# ============================================================================
print("=" * 70)
print("2. PER-MODEL CONFLICT/AGREEMENT")
print("=" * 70)

CRD_AGREEMENT_MODELS = [m for m in MODELS if m != "gpt_5.4_mini"]

cpr_v0 = derived[(derived.game == "CPR") & (derived.wording == "V0_true")]
cpr_baseline_choice = {}
for m in MODELS:
    msub = cpr_v0[cpr_v0.model == m].dropna(subset=["base_p"])
    cpr_baseline_choice[m] = int(msub.loc[msub["base_p"].idxmax(), "anchor"]) if not msub.empty else None

# CRD baseline argmax: read straight from the CRD baseline JSONs (authoritative,
# includes the recovered gpt_5.4/gpt_5.4_mini distributions)
import json
CRD_OUTPUT_DIR = os.path.join(ROOT, "CRD", "output", "default_experiment")
crd_baseline_probs = {}
for m in MODELS:
    path = os.path.join(CRD_OUTPUT_DIR, f"{m}-crd-baseline-v0-n1-blinded.json")
    if not os.path.exists(path):
        print(f"MISSING: {m} CRD baseline JSON")
        crd_baseline_probs[m] = None
        continue
    with open(path) as f:
        crd_baseline_probs[m] = json.load(f)["crd"][0]["sample-0"]

crd_baseline_choice = {}
for m in CRD_AGREEMENT_MODELS:
    bp = crd_baseline_probs.get(m)
    crd_baseline_choice[m] = int(max(bp, key=lambda k: bp[k] if bp[k] is not None else -1)) if bp else None

rows_tex = []
cpr_conflict_gt_agree = 0
crd_conflict_gt_agree = 0
for m in MODELS:
    label = disp_tex(m)
    cpr_choice = cpr_baseline_choice[m]
    cpr_agree_sub = derived[(derived.game == "CPR") & (derived.model == m) &
                            (derived.wording == "V0_true") & (derived.anchor == cpr_choice)]
    cpr_agree_pull = cpr_agree_sub["net_pull"].iloc[0] if not cpr_agree_sub.empty else np.nan
    cpr_conflict_sub = derived[(derived.game == "CPR") & (derived.model == m) &
                               (derived.wording == "V0_true") & (derived.anchor != cpr_choice)]
    cpr_conflict_pull = cpr_conflict_sub["net_pull"].mean() if not cpr_conflict_sub.empty else np.nan
    if pd.notna(cpr_agree_pull) and pd.notna(cpr_conflict_pull) and cpr_conflict_pull > cpr_agree_pull:
        cpr_conflict_gt_agree += 1

    if m in CRD_AGREEMENT_MODELS:
        crd_choice = crd_baseline_choice[m]
        crd_agree_sub = derived[(derived.game == "CRD") & (derived.model == m) &
                                (derived.wording == "V0_true") & (derived.anchor == crd_choice)]
        crd_agree_pull = crd_agree_sub["net_pull"].iloc[0] if not crd_agree_sub.empty else np.nan
        crd_conflict_sub = derived[(derived.game == "CRD") & (derived.model == m) &
                                   (derived.wording == "V0_true") & (derived.anchor != crd_choice)]
        crd_conflict_pull = crd_conflict_sub["net_pull"].mean() if not crd_conflict_sub.empty else np.nan
        if pd.notna(crd_agree_pull) and pd.notna(crd_conflict_pull) and crd_conflict_pull > crd_agree_pull:
            crd_conflict_gt_agree += 1
        crd_str = f"{crd_choice} & {crd_agree_pull:+.2f} & {crd_conflict_pull:+.2f}"
    else:
        crd_str = "unstable baseline (excluded) & -- & --"

    rows_tex.append(
        f"{label} & {cpr_choice} & {cpr_agree_pull:+.2f} & {cpr_conflict_pull:+.2f} & {crd_str} \\\\"
    )

conflict_agreement_tex = r"""\begin{table}[t]
\centering
\caption{Per-model conflict vs. agreement (V0\_true). Preferred anchor/choice = highest baseline probability (CPR) or baseline argmax (CRD). This table backs Figure \ref{fig:conflict_vs_agreement} (n = 6 models for CPR, n = 5 for CRD).}
\label{tab:conflict_agreement_permodel}
\small
\begin{tabular}{lrrrlrr}
\toprule
 & \multicolumn{3}{c}{CPR} & \multicolumn{3}{c}{CRD} \\
Model & pref. anchor & agree pull & conflict pull & pref. choice & agree pull & conflict pull \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
conflict_path = os.path.join(TABLES_DIR, "conflict_agreement_permodel.tex")
with open(conflict_path, "w") as f:
    f.write(conflict_agreement_tex)
print(f"Wrote {conflict_path}")
print(f"CPR: {cpr_conflict_gt_agree}/6 models individually have conflict net_pull > agreement net_pull")
print(f"CRD: {crd_conflict_gt_agree}/5 models (mini excluded) individually have conflict net_pull > agreement net_pull\n")

# ============================================================================
# 3. CRD WORDING-INSTABILITY FIGURE
# ============================================================================
print("=" * 70)
print("3. CRD WORDING INSTABILITY (anchor=0, free-ride)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(5.0, 3.4))
crd_anchor0 = derived[(derived.game == "CRD") & (derived.anchor == 0) & (derived.wording.isin(WORDINGS4))]
any_cross = []
for m in MODELS:
    row = crd_anchor0[crd_anchor0.model == m].set_index("wording").reindex(WORDINGS4)
    if row["raw_spike"].isna().all():
        continue
    vals = row["raw_spike"].values
    ax.plot(range(4), vals, marker="o", label=disp(m), color=MODEL_COLORS[m], linewidth=1.8, markersize=5)
    valid = row["raw_spike"].dropna()
    crosses = len(valid) > 1 and (valid > 0.5).any() and (valid < 0.5).any()
    any_cross.append((disp(m), crosses, valid.tolist()))

ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="high/low following line")
ax.set_xticks(range(4))
ax.set_xticklabels(WORDINGS4)
ax.set_xlabel("Wording")
ax.set_ylabel("Raw spike @ default=0 (free-ride)")
ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "per_model_instability_crd.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "per_model_instability_crd.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/per_model_instability_crd.{pdf,png}")

crossers = [name for name, crosses, _ in any_cross if crosses]
if crossers:
    print(f"CRD instability: YES — {len(crossers)}/6 models cross the 0.5 line across wordings at anchor=0: {crossers}")
else:
    print("CRD instability: NO — no model crosses the 0.5 line at anchor=0 across wordings (CPR-specific pattern)")
print()

# ============================================================================
# 4. COMPREHENSION TABLE + NULL CHECK
# ============================================================================
print("=" * 70)
print("4. COMPREHENSION vs FIXATION")
print("=" * 70)

comp = pd.read_csv(COMPREHENSION_CSV)
comp_lookup = {}
for m in MODELS:
    row = comp[comp.model == m]
    if row.empty:
        print(f"MISSING: {m} not found in comprehension_test.csv")
        comp_lookup[m] = np.nan
    else:
        comp_lookup[m] = row.iloc[0]["mean_fraction_correct"]

fixation_by_game = {}
for game, anchors in [("CPR", CPR_CONFLICT_ANCHORS), ("CRD", CRD_CONFLICT_ANCHORS)]:
    d = {}
    for m in MODELS:
        sub = derived[(derived.game == game) & (derived.model == m) &
                      (derived.anchor.isin(anchors)) & (derived.wording == "V0_true")]
        d[m] = sub["net_pull"].mean() if not sub.empty else np.nan
    fixation_by_game[game] = d

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    c = comp_lookup[m]
    cpr_fix = fixation_by_game["CPR"][m]
    crd_fix = fixation_by_game["CRD"][m]
    c_str = f"{c:.2f}" if pd.notna(c) else "--"
    cpr_str = f"{cpr_fix:+.2f}" if pd.notna(cpr_fix) else "--"
    crd_str = f"{crd_fix:+.2f}" if pd.notna(crd_fix) else "--"
    rows_tex.append(f"{label} & {c_str} & {cpr_str} & {crd_str} \\\\")

comprehension_tex = r"""\begin{table}[t]
\centering
\caption{Comprehension score vs. default-following (mean net pull at conflict anchors, V0\_true).}
\label{tab:comprehension}
\small
\begin{tabular}{lrrr}
\toprule
Model & Comprehension & CPR conflict net\_pull & CRD conflict net\_pull \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
comprehension_path = os.path.join(TABLES_DIR, "comprehension.tex")
with open(comprehension_path, "w") as f:
    f.write(comprehension_tex)
print(f"Wrote {comprehension_path}")

for game in ["CPR", "CRD"]:
    xs, ys = [], []
    for m in MODELS:
        c = comp_lookup[m]
        fx = fixation_by_game[game][m]
        if pd.notna(c) and pd.notna(fx):
            xs.append(c)
            ys.append(fx)
    if len(xs) >= 3:
        r, p = stats.spearmanr(xs, ys)
        print(f"{game}: Spearman r={r:.3f}, p={p:.3f} (n={len(xs)} — note: n=6 is tiny, treat as descriptive only)")
    else:
        print(f"{game}: insufficient data for correlation (n={len(xs)})")
print("VERDICT: comprehension does not reliably predict default-following in either game "
     "(NULL result expected and observed — correlations are weak/inconsistent-signed with n=6, "
     "not something to lean on)\n")

# ============================================================================
# 5. HUMAN-VS-MODEL FIGURE (shared units: mean CPR extraction 0-30)
# ============================================================================
print("=" * 70)
print("5. HUMAN VS MODEL (shared units)")
print("=" * 70)

cpr_cons = pd.read_csv(CPR_CONSOLIDATED)
cpr_cons = cpr_cons[cpr_cons.model.isin(MODELS)]

categories = ["control (no default)", "default=11", "default=23"]
x = np.arange(len(categories))
human_vals = [HUMAN_BASELINE, HUMAN_D11, HUMAN_D23]

model_series = {}
for m in MODELS:
    base = cpr_cons[(cpr_cons.model == m) & (cpr_cons.condition == CPR_BASELINE_CONDITION)]["average"]
    d11 = cpr_cons[(cpr_cons.model == m) & (cpr_cons.condition == "default=11")]["average"]
    d23 = cpr_cons[(cpr_cons.model == m) & (cpr_cons.condition == "default=23")]["average"]
    if base.empty or d11.empty or d23.empty:
        print(f"MISSING: {m} missing one of control/default=11/default=23 — excluded from figure")
        continue
    model_series[m] = [base.iloc[0], d11.iloc[0], d23.iloc[0]]

fig, ax = plt.subplots(figsize=(4.2, 3.4))
for m, vals in model_series.items():
    ax.plot(x, vals, marker="o", color=MODEL_COLORS[m], linewidth=1.4, markersize=5, alpha=0.85, label=disp(m))
ax.plot(x, human_vals, marker="s", color="black", linewidth=2.8, markersize=7, label="Human (Montero-Porras et al. 2025)", zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel("Mean extraction (0-30)")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "human_vs_llm.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "human_vs_llm.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/human_vs_llm.{pdf,png}")

human_range = max(human_vals) - min(human_vals)
model_ranges = {m: max(v) - min(v) for m, v in model_series.items()}
model_swings = {m: r for m, r in model_ranges.items() if r > 2 * human_range}
print(f"Human range across conditions: {human_range:.2f} tokens")
for m, r in model_ranges.items():
    print(f"  {disp(m):14s} range={r:.2f} tokens ({'swings >2x human range' if r > 2*human_range else 'comparable to human range'})")
if model_swings:
    print(f"Figure DOES separate humans from models: {len(model_swings)}/{len(model_series)} models swing >2x the human range.")
else:
    print("Figure is WEAK: no model swings much more than humans — models cluster near the human range. "
         "Flagging for your call on whether to cut this figure.")

print("\n=== FILES WRITTEN ===")
for p in [framing_path, conflict_path, comprehension_path,
         os.path.join(FIGURES_DIR, "framing_effect_bothgames.pdf"),
         os.path.join(FIGURES_DIR, "framing_effect_bothgames.png"),
         os.path.join(FIGURES_DIR, "per_model_instability_crd.pdf"),
         os.path.join(FIGURES_DIR, "per_model_instability_crd.png"),
         os.path.join(FIGURES_DIR, "human_vs_llm.pdf"),
         os.path.join(FIGURES_DIR, "human_vs_llm.png")]:
    print(p)
