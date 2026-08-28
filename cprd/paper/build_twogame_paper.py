"""Publication-ready tables/figures for the two-game default-framing paper
(CPR extract-from-pool + CRD contribute-to-threshold). Pure post-processing,
no API calls. Baseline-corrects every spike into net_pull, the primary
quantity reported everywhere except the wording-robustness table (which
also needs raw_spike to talk about the 0.5 fixator/rejector line).

Reads (read-only):
  - cprd/output/mult_default/consolidated.csv           (CPR main run, V0_true)
  - cprd/output/prompt_robustness/summary.csv            (CPR wording robustness, anchors 18/26)
  - CRD/output/default_experiment/summary.csv            (CRD spikes)
  - CRD/output/default_experiment/*-baseline-*.json      (CRD per-token baseline probs)
Hardcoded: human benchmark (Montero-Porras et al. 2025, Table 1).

Writes:
  - cprd/paper/derived_metrics_twogame.csv
  - cprd/paper/tables/{design,crd_spike,cpr_spike,wording_robustness}.tex
  - cprd/paper/figures/{framing_effect_bothgames,conflict_vs_agreement,
                        per_model_instability,human_vs_llm}.{pdf,png}
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # LLMs/
CPRD_DIR = os.path.join(ROOT, "cprd")
CRD_DIR = os.path.join(ROOT, "CRD")
PAPER_DIR = os.path.join(CPRD_DIR, "paper")
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CPR_CONSOLIDATED = os.path.join(CPRD_DIR, "output", "mult_default", "consolidated.csv")
CPR_ROBUSTNESS = os.path.join(CPRD_DIR, "output", "prompt_robustness", "summary.csv")
CRD_SUMMARY = os.path.join(CRD_DIR, "output", "default_experiment", "summary.csv")
CRD_OUTPUT_DIR = os.path.join(CRD_DIR, "output", "default_experiment")

CPR_BASELINE_CONDITION = "no default, no group extraction"
CPR_ANCHORS = [0, 11, 18, 23, 26, 30]
CPR_ROBUSTNESS_ANCHORS = [18, 26]
CPR_ROBUSTNESS_WORDINGS = ["V1", "V2", "V3", "V4"]  # V0_true already covered by the main-run block
CRD_ANCHORS = [0, 2, 4]
CRD_WORDINGS = ["V0_true", "V1", "V2", "V3"]

MODELS = [
    "gpt_5.1", "gpt_5.4", "gpt_5.4_mini", "gpt_5.4_nano",
    "Llama_3.3_70B_Instruct", "Qwen3_235B_A22B_Instruct",
]
DISPLAY_NAMES = {
    "gpt_5.1": "gpt_5.1", "gpt_5.4": "gpt_5.4", "gpt_5.4_mini": "gpt_5.4_mini",
    "gpt_5.4_nano": "gpt_5.4_nano", "Llama_3.3_70B_Instruct": "Llama3.3",
    "Qwen3_235B_A22B_Instruct": "Qwen3",
}


def disp(m: str) -> str:
    return DISPLAY_NAMES.get(m, m.replace("_", " "))


def disp_tex(m: str) -> str:
    return disp(m).replace("_", r"\_")


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
MODEL_COLORS = {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(MODELS)}

HUMAN_BASELINE = 16.32
HUMAN_D11 = 15.72
HUMAN_D23 = 17.84

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

# ----------------------------------------------------------------------------
# SANITY GATE: Qwen CPR baseline
# ----------------------------------------------------------------------------
cpr_df = pd.read_csv(CPR_CONSOLIDATED)
cpr_df_sub = cpr_df[cpr_df["model"].isin(MODELS)].copy()

qwen_base_row = cpr_df_sub[(cpr_df_sub.model == "Qwen3_235B_A22B_Instruct") &
                           (cpr_df_sub.condition == CPR_BASELINE_CONDITION)]
if qwen_base_row.empty or pd.isna(qwen_base_row.iloc[0]["average"]):
    print("MISSING: Qwen3_235B_A22B_Instruct CPR baseline average — cannot run sanity gate. Aborting.")
    sys.exit(1)
qwen_base_avg = qwen_base_row.iloc[0]["average"]
print(f"SANITY GATE: Qwen3_235B_A22B_Instruct CPR baseline average = {qwen_base_avg:.3f}")
if qwen_base_avg < 5:
    print("STOP: stale Qwen (baseline < 5). Aborting, nothing written.")
    sys.exit(1)
print("Sanity gate PASSED (Qwen CPR baseline >= 5).\n")

# ----------------------------------------------------------------------------
# CPR baselines (per model, p_d for every default value + full row)
# ----------------------------------------------------------------------------
cpr_baseline_rows = {}
print("=== CPR per-model baseline (average, argmax reading) ===")
for m in MODELS:
    row = cpr_df_sub[(cpr_df_sub.model == m) & (cpr_df_sub.condition == CPR_BASELINE_CONDITION)]
    if row.empty:
        print(f"MISSING: {m} CPR baseline — no row found")
        cpr_baseline_rows[m] = None
        continue
    cpr_baseline_rows[m] = row.iloc[0]
    print(f"  {disp(m):14s} baseline average={row.iloc[0]['average']:.2f}")
print()

# ----------------------------------------------------------------------------
# CRD baselines (per model, per-token p from the baseline JSON)
# ----------------------------------------------------------------------------
crd_baseline_probs = {}  # model -> {"0":p, "2":p, "4":p}
print("=== CRD per-model baseline (argmax, prob) ===")
for m in MODELS:
    path = os.path.join(CRD_OUTPUT_DIR, f"{m}-crd-baseline-v0-n1-blinded.json")
    if not os.path.exists(path):
        print(f"MISSING: {m} CRD baseline JSON not found at {path}")
        crd_baseline_probs[m] = None
        continue
    with open(path) as f:
        d = json.load(f)
    probs = d["crd"][0]["sample-0"]
    crd_baseline_probs[m] = probs
    print(f"  {disp(m):14s} argmax={d.get('argmax')} p={d.get('argmax_prob'):.3f}  "
         f"(p0={probs.get('0')}, p2={probs.get('2')}, p4={probs.get('4')})")
print()

# ----------------------------------------------------------------------------
# Build derived_metrics_twogame.csv
# ----------------------------------------------------------------------------
records = []

# --- CPR: main 6 anchors, wording=V0_true, from consolidated.csv ---
p_cols_cpr = [f"p_{i}" for i in range(31)]
for m in MODELS:
    brow = cpr_baseline_rows[m]
    for d in CPR_ANCHORS:
        arow_df = cpr_df_sub[(cpr_df_sub.model == m) & (cpr_df_sub.condition == f"default={d}")]
        if arow_df.empty:
            print(f"MISSING: CPR {m} default={d} — no row found")
            continue
        arow = arow_df.iloc[0]
        pcol = f"p_{d}"
        raw_spike = arow.get(pcol)
        footnote = False
        if pd.isna(raw_spike):
            raw_spike = 0.0
            footnote = True
        base_p = brow[pcol] if brow is not None and pcol in brow.index else None
        if base_p is None or pd.isna(base_p):
            print(f"MISSING: CPR {m} baseline p_{d} — skipping net_pull for this cell")
            continue
        net_pull = raw_spike - base_p
        coverage = arow[p_cols_cpr].astype(float).sum() if all(c in arow.index for c in p_cols_cpr) else np.nan
        records.append({
            "game": "CPR", "model": m, "anchor": d, "wording": "V0_true",
            "raw_spike": raw_spike, "base_p": base_p, "net_pull": net_pull,
            "coverage": coverage, "footnote_unresolved": footnote,
        })

# --- CPR: wording robustness, anchors 18/26, V1-V4, from prompt_robustness/summary.csv ---
cpr_rob = pd.read_csv(CPR_ROBUSTNESS)
cpr_rob = cpr_rob[cpr_rob["model"].isin(MODELS) & cpr_rob["default"].isin(CPR_ROBUSTNESS_ANCHORS) &
                  cpr_rob["variant"].isin(CPR_ROBUSTNESS_WORDINGS)]
for _, r in cpr_rob.iterrows():
    m, d, wording = r["model"], int(r["default"]), r["variant"]
    if pd.notna(r.get("error")) and str(r.get("error")).strip():
        print(f"MISSING: CPR {m} default={d} {wording} — FAILED cell ({r['error']}), skipping")
        continue
    raw_spike = r["spike"]
    footnote = False
    if pd.isna(raw_spike):
        raw_spike = 0.0
        footnote = True
    brow = cpr_baseline_rows[m]
    pcol = f"p_{d}"
    base_p = brow[pcol] if brow is not None and pcol in brow.index else None
    if base_p is None or pd.isna(base_p):
        print(f"MISSING: CPR {m} baseline p_{d} — skipping net_pull for {wording}")
        continue
    net_pull = raw_spike - base_p
    records.append({
        "game": "CPR", "model": m, "anchor": d, "wording": wording,
        "raw_spike": raw_spike, "base_p": base_p, "net_pull": net_pull,
        "coverage": r.get("coverage"), "footnote_unresolved": footnote,
    })

# --- CRD: anchors 0/2/4, V0_true..V3, from CRD summary.csv + baseline JSONs ---
crd_summary = pd.read_csv(CRD_SUMMARY)
crd_summary = crd_summary[crd_summary["model"].isin(MODELS) & (crd_summary["condition"] != "baseline")]
for _, r in crd_summary.iterrows():
    m, d, wording = r["model"], int(r["condition"]), r["variant"]
    if d not in CRD_ANCHORS or wording not in CRD_WORDINGS:
        continue
    if pd.notna(r.get("error")) and str(r.get("error")).strip():
        print(f"MISSING: CRD {m} default={d} {wording} — FAILED cell ({r['error']}), skipping")
        continue
    raw_spike = r["spike"]
    footnote = False
    if pd.isna(raw_spike):
        raw_spike = 0.0
        footnote = True
    bp = crd_baseline_probs.get(m)
    if bp is None:
        print(f"MISSING: CRD {m} baseline probs — skipping net_pull for default={d} {wording}")
        continue
    base_p = bp.get(str(d))
    if base_p is None:
        print(f"MISSING: CRD {m} baseline p({d}) — skipping net_pull for {wording}")
        continue
    net_pull = raw_spike - base_p
    records.append({
        "game": "CRD", "model": m, "anchor": d, "wording": wording,
        "raw_spike": raw_spike, "base_p": base_p, "net_pull": net_pull,
        "coverage": r.get("coverage"), "footnote_unresolved": footnote,
    })

derived = pd.DataFrame.from_records(records)
derived_path = os.path.join(PAPER_DIR, "derived_metrics_twogame.csv")
derived.to_csv(derived_path, index=False)
print(f"\nWrote {derived_path} ({len(derived)} rows)\n")

# ----------------------------------------------------------------------------
# Conflict-anchor net_pull by wording, printed for both games (text sanity check)
# ----------------------------------------------------------------------------
print("=== CONFLICT-ANCHOR mean net_pull by wording (sanity check before trusting figures) ===")
print("CPR (anchors 18, 26):")
cpr_conflict = derived[(derived.game == "CPR") & (derived.anchor.isin([18, 26]))]
for w in ["V0_true", "V1", "V2", "V3"]:
    sub = cpr_conflict[cpr_conflict.wording == w]
    print(f"  {w}: mean net_pull = {sub.net_pull.mean():.3f}  (n={len(sub)})")
print("CRD (conflict anchors 0, 2 — anchor 4 matches baseline preference for most models):")
crd_conflict = derived[(derived.game == "CRD") & (derived.anchor.isin([0, 2]))]
for w in CRD_WORDINGS:
    sub = crd_conflict[crd_conflict.wording == w]
    print(f"  {w}: mean net_pull = {sub.net_pull.mean():.3f}  (n={len(sub)})")
print()

# ----------------------------------------------------------------------------
# TABLE design.tex (both games, models + conditions)
# ----------------------------------------------------------------------------
MODEL_META = {
    "gpt_5.1": {"provider": "OpenAI", "params": "undisclosed"},
    "gpt_5.4": {"provider": "OpenAI", "params": "undisclosed"},
    "gpt_5.4_mini": {"provider": "OpenAI", "params": "undisclosed"},
    "gpt_5.4_nano": {"provider": "OpenAI", "params": "undisclosed"},
    "Llama_3.3_70B_Instruct": {"provider": "Meta", "params": "70B"},
    "Qwen3_235B_A22B_Instruct": {"provider": "Alibaba (Qwen team)", "params": "235B (22B active)"},
}
model_lines = [f"{disp_tex(m)} & {MODEL_META[m]['provider']} & {MODEL_META[m]['params']} & logprob \\\\" for m in MODELS]

cpr_condition_rows = [
    ("baseline / none", "none"), ("default = 0", "abstain"), ("default = 11", "social optimum"),
    ("default = 18", "Nash equilibrium"), ("default = 23", "exploitative"),
    ("default = 26", "indefensible"), ("default = 30", "collapse"),
]
crd_condition_rows = [
    ("baseline / none", "none"), ("default = 0", "free-ride (selfish)"),
    ("default = 2", "middle"), ("default = 4", "full cooperation (safe)"),
]
cpr_cond_lines = [f"{lbl} & {interp} \\\\" for lbl, interp in cpr_condition_rows]
crd_cond_lines = [f"{lbl} & {interp} \\\\" for lbl, interp in crd_condition_rows]

design_tex = r"""\begin{table}[t]
\centering
\caption{Experimental design: models and conditions, both games.}
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

\begin{minipage}{0.48\linewidth}
\centering
\small
\begin{tabular}{ll}
\toprule
CPR condition & Benchmark interp. \\
\midrule
""" + "\n".join(cpr_cond_lines) + r"""
\bottomrule
\end{tabular}
\end{minipage}
\hfill
\begin{minipage}{0.48\linewidth}
\centering
\small
\begin{tabular}{ll}
\toprule
CRD condition & Benchmark interp. \\
\midrule
""" + "\n".join(crd_cond_lines) + r"""
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
# TABLE 1: crd_spike.tex (net_pull, anchor x wording, + baseline argmax col)
# ----------------------------------------------------------------------------
crd_pivot = derived[derived.game == "CRD"].pivot_table(
    index="model", columns=["anchor", "wording"], values="net_pull", aggfunc="first")
crd_pivot = crd_pivot.reindex(index=MODELS, columns=pd.MultiIndex.from_product([CRD_ANCHORS, CRD_WORDINGS]))

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    bp = crd_baseline_probs.get(m)
    base_argmax = "--"
    if bp is not None:
        base_argmax = max(bp, key=lambda k: bp[k] if bp[k] is not None else -1)
    vals = []
    for a in CRD_ANCHORS:
        for w in CRD_WORDINGS:
            v = crd_pivot.loc[m, (a, w)]
            vals.append(f"{v:+.2f}" if pd.notna(v) else "--")
    rows_tex.append(f"{label} & " + " & ".join(vals) + f" & {base_argmax}" + r" \\")

colmean = crd_pivot.mean(axis=0, skipna=True)
colmean_row = "column mean & " + " & ".join(
    f"{colmean[(a, w)]:+.2f}" if pd.notna(colmean[(a, w)]) else "--" for a in CRD_ANCHORS for w in CRD_WORDINGS
) + " & \\\\"

crd_tex = r"""\begin{table}[t]
\centering
\caption{CRD: baseline-corrected net pull toward each pre-filled contribution (net\_pull = raw spike $-$ baseline probability of that value), by anchor and wording. Positive = default adds probability mass beyond the model's own baseline preference.}
\label{tab:crd_spike}
\small
\begin{tabular}{lrrrrrrrrrrrrl}
\toprule
Model & \multicolumn{4}{c}{$d{=}0$} & \multicolumn{4}{c}{$d{=}2$} & \multicolumn{4}{c}{$d{=}4$} & baseline \\
 & V0t & V1 & V2 & V3 & V0t & V1 & V2 & V3 & V0t & V1 & V2 & V3 & argmax \\
\midrule
""" + "\n".join(rows_tex) + r"""
\midrule
""" + colmean_row + r"""
\bottomrule
\end{tabular}
\end{table}
"""
crd_path = os.path.join(TABLES_DIR, "crd_spike.tex")
with open(crd_path, "w") as f:
    f.write(crd_tex)
print(f"Wrote {crd_path}")

# ----------------------------------------------------------------------------
# TABLE 2: cpr_spike.tex (net_pull, 6 anchors, V0_true only)
# ----------------------------------------------------------------------------
cpr_v0 = derived[(derived.game == "CPR") & (derived.wording == "V0_true")]
cpr_v0_pivot = cpr_v0.pivot_table(index="model", columns="anchor", values="net_pull", aggfunc="first")
cpr_v0_pivot = cpr_v0_pivot.reindex(index=MODELS, columns=CPR_ANCHORS)

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    vals = [f"{cpr_v0_pivot.loc[m, d]:+.2f}" if pd.notna(cpr_v0_pivot.loc[m, d]) else "--" for d in CPR_ANCHORS]
    rows_tex.append(f"{label} & " + " & ".join(vals) + r" \\")
colmean = cpr_v0_pivot.mean(axis=0, skipna=True)
colmean_row = "column mean & " + " & ".join(f"{colmean[d]:+.2f}" if pd.notna(colmean[d]) else "--" for d in CPR_ANCHORS) + r" \\"

cpr_tex = r"""\begin{table}[t]
\centering
\caption{CPR: baseline-corrected net pull toward each default (net\_pull = raw spike $-$ baseline probability of that value), main-run wording (V0\_true) only.}
\label{tab:cpr_spike}
\small
\begin{tabular}{lrrrrrr}
\toprule
Model & $d{=}0$ & $d{=}11$ & $d{=}18$ & $d{=}23$ & $d{=}26$ & $d{=}30$ \\
\midrule
""" + "\n".join(rows_tex) + r"""
\midrule
""" + colmean_row + r"""
\bottomrule
\end{tabular}
\end{table}
"""
cpr_path = os.path.join(TABLES_DIR, "cpr_spike.tex")
with open(cpr_path, "w") as f:
    f.write(cpr_tex)
print(f"Wrote {cpr_path}")

# ----------------------------------------------------------------------------
# TABLE 3: wording_robustness.tex (CPR anchors 18/26, raw_spike, V0_true..V3)
# ----------------------------------------------------------------------------
rob_wordings = ["V0_true", "V1", "V2", "V3"]
rob = derived[(derived.game == "CPR") & (derived.anchor.isin([18, 26])) & (derived.wording.isin(rob_wordings))]
rob_pivot = rob.pivot_table(index=["model", "anchor"], columns="wording", values="raw_spike", aggfunc="first")

crossing_notes = []
rows_tex = []
for d in [18, 26]:
    for m in MODELS:
        key = (m, d)
        if key not in rob_pivot.index:
            continue
        row = rob_pivot.loc[key].reindex(rob_wordings)
        vals = [f"{row[w]:.2f}" if pd.notna(row[w]) else "--" for w in rob_wordings]
        crosses = row.dropna()
        if len(crosses) > 1 and ((crosses > 0.5).any() and (crosses < 0.5).any()):
            crossing_notes.append(f"{disp(m)}@{d}")
        label = f"{disp_tex(m)} ($d{{=}}{d}$)"
        rows_tex.append(f"{label} & " + " & ".join(vals) + r" \\")

crossing_note_str = ", ".join(crossing_notes) if crossing_notes else "none"
wording_tex = r"""\begin{table}[t]
\centering
\caption{CPR wording robustness at the two load-bearing anchors (raw spike, not baseline-corrected). Cells crossing the 0.5 fixator/rejector line across wordings: """ + crossing_note_str + r"""}
\label{tab:wording_robustness}
\small
\begin{tabular}{lrrrr}
\toprule
Model (anchor) & V0\_true & V1 & V2 & V3 \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
wording_path = os.path.join(TABLES_DIR, "wording_robustness.tex")
with open(wording_path, "w") as f:
    f.write(wording_tex)
print(f"Wrote {wording_path}")

# ----------------------------------------------------------------------------
# FIGURE 1: framing_effect_bothgames.pdf (headline, two panels)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)

cpr_anchors_fig1 = [18, 26]
ax = axes[0]
for d in cpr_anchors_fig1:
    sub = derived[(derived.game == "CPR") & (derived.anchor == d) & (derived.wording.isin(["V0_true", "V1", "V2", "V3"]))]
    means = sub.groupby("wording")["net_pull"].mean().reindex(["V0_true", "V1", "V2", "V3"])
    ax.plot(range(4), means.values, marker="o", label=f"d={d}", linewidth=1.8, markersize=5)
ax.set_xticks(range(4))
ax.set_xticklabels(["V0t", "V1", "V2", "V3"])
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Mean net pull (across models)")
ax.set_xlabel("Wording")
ax.set_title("CPR (extract)", fontsize=10)
ax.legend(fontsize=8)

crd_anchors_fig1 = [0, 2]
ax = axes[1]
for d in crd_anchors_fig1:
    sub = derived[(derived.game == "CRD") & (derived.anchor == d) & (derived.wording.isin(["V0_true", "V1", "V2", "V3"]))]
    means = sub.groupby("wording")["net_pull"].mean().reindex(["V0_true", "V1", "V2", "V3"])
    ax.plot(range(4), means.values, marker="s", label=f"d={d}", linewidth=1.8, markersize=5)
ax.set_xticks(range(4))
ax.set_xticklabels(["V0t", "V1", "V2", "V3"])
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Wording")
ax.set_title("CRD (contribute)", fontsize=10)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "framing_effect_bothgames.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "framing_effect_bothgames.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/framing_effect_bothgames.{pdf,png}")

# ----------------------------------------------------------------------------
# FIGURE 2: conflict_vs_agreement.pdf (per-model AGREEMENT vs CONFLICT, both games)
# ----------------------------------------------------------------------------
# baseline choice per model per game:
#   CPR: the anchor with the HIGHEST baseline base_p (not nearest-to-mean -
#        nearest-to-mean was a poor proxy: base_p at the "nearest" anchor of
#        a diffuse 31-way distribution is generically small even when close
#        to the mean, which inflated "agreement" net_pull artificially).
#   CRD: argmax over {0,2,4} from the (now-recovered) baseline JSON.
# CRD excludes gpt_5.4_mini: its two independent baseline calls landed on
# opposite argmax (0 vs 4) at matched confidence (~0.62 either way) - a
# genuine coin-flip, not a stable baseline preference, so it can't be
# assigned a meaningful AGREEMENT/CONFLICT label.
CRD_AGREEMENT_MODELS = [m for m in MODELS if m != "gpt_5.4_mini"]

cpr_v0_only = derived[(derived.game == "CPR") & (derived.wording == "V0_true")]
cpr_baseline_choice = {}
for m in MODELS:
    msub = cpr_v0_only[cpr_v0_only.model == m].dropna(subset=["base_p"])
    if msub.empty:
        cpr_baseline_choice[m] = None
        continue
    cpr_baseline_choice[m] = int(msub.loc[msub["base_p"].idxmax(), "anchor"])

crd_baseline_choice = {}
for m in CRD_AGREEMENT_MODELS:
    bp = crd_baseline_probs.get(m)
    if bp is None:
        crd_baseline_choice[m] = None
        continue
    crd_baseline_choice[m] = int(max(bp, key=lambda k: bp[k] if bp[k] is not None else -1))

print("=== Baseline choice per model (used for AGREEMENT/CONFLICT labeling) ===")
for m in MODELS:
    crd_choice = crd_baseline_choice.get(m, "EXCLUDED (unstable baseline)")
    print(f"  {disp(m):14s} CPR highest-base_p-anchor={cpr_baseline_choice[m]}   CRD argmax={crd_choice}")
print()

def label_agreement(game, model, anchor):
    if game == "CRD" and model not in CRD_AGREEMENT_MODELS:
        return None
    choice = cpr_baseline_choice[model] if game == "CPR" else crd_baseline_choice[model]
    if choice is None:
        return None
    return "AGREEMENT" if anchor == choice else "CONFLICT"

derived["agreement"] = derived.apply(lambda r: label_agreement(r["game"], r["model"], r["anchor"]), axis=1)

# use V0_true only, so the comparison isn't diluted/duplicated across wordings
agree_sub = derived[(derived.wording == "V0_true") & derived.agreement.notna()]

print("=== AGREEMENT vs CONFLICT mean net_pull, PER GAME (V0_true) — no pooled verdict ===")
game_stats = {}
for game in ["CPR", "CRD"]:
    gsub = agree_sub[agree_sub.game == game]
    stats = gsub.groupby("agreement")["net_pull"].agg(["mean", "count"])
    a_mean = stats.loc["AGREEMENT", "mean"] if "AGREEMENT" in stats.index else float("nan")
    a_n = int(stats.loc["AGREEMENT", "count"]) if "AGREEMENT" in stats.index else 0
    c_mean = stats.loc["CONFLICT", "mean"] if "CONFLICT" in stats.index else float("nan")
    c_n = int(stats.loc["CONFLICT", "count"]) if "CONFLICT" in stats.index else 0
    game_stats[game] = {"a_mean": a_mean, "a_n": a_n, "c_mean": c_mean, "c_n": c_n}
    print(f"  {game}: Agreement mean={a_mean:.3f} (n={a_n})   Conflict mean={c_mean:.3f} (n={c_n})")
    if a_mean < 0.10 and c_mean > a_mean:
        game_verdict = f"MECHANISM SUPPORTED (agreement={a_mean:.3f}, conflict={c_mean:.3f})"
    else:
        game_verdict = f"MECHANISM WEAK (agreement={a_mean:.3f}, conflict={c_mean:.3f})"
    game_stats[game]["verdict"] = game_verdict
    print(f"    -> {game}: {game_verdict}")
print()

print("=== Per-model: baseline choice, net_pull at own anchor vs mean at other anchors (V0_true) ===")
for game in ["CPR", "CRD"]:
    print(f"  -- {game} --")
    choice_map = cpr_baseline_choice if game == "CPR" else crd_baseline_choice
    model_list = MODELS if game == "CPR" else CRD_AGREEMENT_MODELS
    for m in model_list:
        msub = agree_sub[(agree_sub.game == game) & (agree_sub.model == m)]
        if msub.empty:
            continue
        own = msub[msub.agreement == "AGREEMENT"]["net_pull"]
        other = msub[msub.agreement == "CONFLICT"]["net_pull"]
        own_str = f"{own.mean():.3f}" if len(own) else "n/a"
        other_str = f"{other.mean():.3f}" if len(other) else "n/a"
        print(f"    {disp(m):14s} baseline_choice={choice_map[m]}  own_anchor_net_pull={own_str}  other_anchors_mean={other_str}")
    if game == "CRD":
        print(f"    (gpt_5.4_mini excluded: unstable coin-flip baseline)")
print()

fig, ax = plt.subplots(figsize=(5.8, 3.8))
bar_data = []
for game in ["CPR", "CRD"]:
    s = game_stats[game]
    bar_data.append((game, "AGREEMENT", f"{game}\nAgreement\n(n={s['a_n']})", s["a_mean"], "#009E73"))
    bar_data.append((game, "CONFLICT", f"{game}\nConflict\n(n={s['c_n']})", s["c_mean"], "#D55E00"))
xs = np.arange(len(bar_data))
vals = [b[3] for b in bar_data]
colors = [b[4] for b in bar_data]
ax.bar(xs, vals, color=colors, alpha=0.35, edgecolor="black", linewidth=0.5, width=0.65, zorder=1)

# overlay each contributing model as its own jittered, coloured dot so it's
# visible which models drive each bar, not just the aggregate mean
rng = np.random.default_rng(0)
plotted_models = set()
for i, (game, label, xticklbl, v, c) in enumerate(bar_data):
    gsub = agree_sub[(agree_sub.game == game) & (agree_sub.agreement == label)]
    for _, row in gsub.iterrows():
        jitter = rng.uniform(-0.18, 0.18)
        ax.scatter(i + jitter, row["net_pull"], color=MODEL_COLORS[row["model"]],
                  edgecolor="black", linewidth=0.4, s=32, zorder=3,
                  label=disp(row["model"]) if row["model"] not in plotted_models else None)
        plotted_models.add(row["model"])
    ax.text(i, v + (0.05 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=8.5,
           fontweight="bold", zorder=4)

ax.axhline(0, color="gray", linewidth=0.8, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels([b[2] for b in bar_data], fontsize=8)
ax.set_ylabel("Net pull (V0_true)")
handles, labels = ax.get_legend_handles_labels()
disp_order = [disp(m) for m in MODELS]
order = [disp_order.index(l) if l in disp_order else 99 for l in labels]
# reorder legend to match MODELS order rather than first-appearance order
paired_idx = sorted(range(len(order)), key=lambda i: order[i])
handles = [handles[i] for i in paired_idx]
labels = [labels[i] for i in paired_idx]
ax.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), title="model", title_fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "conflict_vs_agreement.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "conflict_vs_agreement.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/conflict_vs_agreement.{pdf,png}")

# ----------------------------------------------------------------------------
# FIGURE 3: per_model_instability.pdf (CPR anchor=26, raw_spike by wording)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 3.4))
sub26 = derived[(derived.game == "CPR") & (derived.anchor == 26) & (derived.wording.isin(["V0_true", "V1", "V2", "V3"]))]
for m in MODELS:
    row = sub26[sub26.model == m].set_index("wording").reindex(["V0_true", "V1", "V2", "V3"])
    if row["raw_spike"].isna().all():
        continue
    ax.plot(range(4), row["raw_spike"].values, marker="o", label=disp(m),
           color=MODEL_COLORS[m], linewidth=1.8, markersize=5)
ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="fixator/rejector line")
ax.set_xticks(range(4))
ax.set_xticklabels(["V0_true", "V1", "V2", "V3"])
ax.set_xlabel("Wording")
ax.set_ylabel("Raw spike @ default=26")
ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "per_model_instability.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "per_model_instability.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/per_model_instability.{pdf,png}")

# ----------------------------------------------------------------------------
# FIGURE 4: human_vs_llm.pdf (CPR, net_pull band vs human)
# ----------------------------------------------------------------------------
categories = ["baseline", "default=11", "default=23"]
x = np.arange(len(categories))
human_vals = [HUMAN_BASELINE, HUMAN_D11, HUMAN_D23]

llm_base_avgs = [cpr_baseline_rows[m]["average"] for m in MODELS if cpr_baseline_rows[m] is not None]
llm_d11 = cpr_df_sub[(cpr_df_sub.model.isin(MODELS)) & (cpr_df_sub.condition == "default=11")]["average"].dropna().tolist()
llm_d23 = cpr_df_sub[(cpr_df_sub.model.isin(MODELS)) & (cpr_df_sub.condition == "default=23")]["average"].dropna().tolist()

llm_means = [np.mean(llm_base_avgs), np.mean(llm_d11), np.mean(llm_d23)]
llm_mins = [np.min(llm_base_avgs), np.min(llm_d11), np.min(llm_d23)]
llm_maxs = [np.max(llm_base_avgs), np.max(llm_d11), np.max(llm_d23)]

fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.fill_between(x, llm_mins, llm_maxs, color="#0072B2", alpha=0.2, label="LLM mean (range)")
ax.plot(x, llm_means, marker="o", color="#0072B2", linewidth=1.8, markersize=6, label="LLM mean (range)")
ax.plot(x, human_vals, marker="s", color="black", linewidth=2.4, markersize=6,
       label="Human (Montero-Porras et al. 2025)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Extraction")
handles, labels = ax.get_legend_handles_labels()
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
# FINAL SUMMARY
# ----------------------------------------------------------------------------
print("\n=== FILES WRITTEN ===")
for p in [
    derived_path, design_path, crd_path, cpr_path, wording_path,
    os.path.join(FIGURES_DIR, "framing_effect_bothgames.pdf"),
    os.path.join(FIGURES_DIR, "framing_effect_bothgames.png"),
    os.path.join(FIGURES_DIR, "conflict_vs_agreement.pdf"),
    os.path.join(FIGURES_DIR, "conflict_vs_agreement.png"),
    os.path.join(FIGURES_DIR, "per_model_instability.pdf"),
    os.path.join(FIGURES_DIR, "per_model_instability.png"),
    os.path.join(FIGURES_DIR, "human_vs_llm.pdf"),
    os.path.join(FIGURES_DIR, "human_vs_llm.png"),
]:
    print(p)
