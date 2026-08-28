"""Distance-aware default-influence metric (normalized Wasserstein-to-default,
w_pull) alongside the existing exact-match net_pull, computed from the FULL
per-choice distributions already stored on disk. No model calls - pure
post-processing/re-derivation.

w_pull = 1 - W(D, P_def) / W(D, P_base)
  D      = degenerate distribution at the default value d
  P_base = model's no-default baseline distribution over the action space
  P_def  = model's distribution under the default condition
  W      = 1-Wasserstein (earth-mover) distance on the ordered value axis,
           which for a degenerate D collapses to the weighted mean absolute
           distance from d: W(D, P) = sum_i P(x_i) * |x_i - d|.
The ratio W_def/W_base is already scale-free (same units cancel), so CPR
(0-30) and CRD ({0,2,4}) w_pull values are directly comparable without
extra normalization - confirmed by construction, not assumed.

Reads (read-only), full distributions not just scalars:
  - cprd/output/mult_default/consolidated.csv                (CPR V0_true, 6 anchors + baseline)
  - cprd/output/prompt_robustness/*.json                      (CPR V1-V4, anchors 18/26; skips
                                                                _confounded_v0_no_leadin/)
  - CRD/output/default_experiment/*.json                      (CRD V0_true-V3, anchors 0/2/4,
                                                                and baselines; skips
                                                                _baseline_pre_recovery/)
  - cprd/output/comprehension_test.csv, CRD/.../crd_comprehension.csv
Hardcoded: none new (no human benchmark needed for these figures).

Writes (all under cprd/paper_wpull/, a separate folder from cprd/paper/ so the
existing net_pull-only assets are untouched):
  - derived_metrics_twogame_v2.csv
  - tables/{framing_permodel,conflict_agreement_permodel,comprehension}_wpull.tex
  - figures/{framing_effect_bothgames,conflict_vs_agreement}_wpull.{pdf,png}
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # LLMs/
CPRD_DIR = os.path.join(ROOT, "cprd")
CRD_DIR = os.path.join(ROOT, "CRD")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # cprd/paper_wpull/
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CPR_CONSOLIDATED = os.path.join(CPRD_DIR, "output", "mult_default", "consolidated.csv")
CPR_ROBUSTNESS_DIR = os.path.join(CPRD_DIR, "output", "prompt_robustness")
CRD_OUTPUT_DIR = os.path.join(CRD_DIR, "output", "default_experiment")
CPR_COMPREHENSION_CSV = os.path.join(CPRD_DIR, "output", "comprehension_test.csv")
CRD_COMPREHENSION_CSV = os.path.join(CRD_DIR, "output", "default_experiment", "crd_comprehension.csv")

CPR_BASELINE_CONDITION = "no default, no group extraction"
CPR_ANCHORS = [0, 11, 18, 23, 26, 30]
CPR_ROBUSTNESS_ANCHORS = [18, 26]
CPR_ROBUSTNESS_WORDINGS = ["V1", "V2", "V3", "V4"]
CRD_ANCHORS = [0, 2, 4]
CRD_WORDINGS = ["V0_true", "V1", "V2", "V3"]
CPR_CONFLICT_ANCHORS = [18, 26]
CRD_CONFLICT_ANCHORS = [0, 2]
WORDINGS4 = ["V0_true", "V1", "V2", "V3"]
COVERAGE_MIN = 0.90

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

plt.rcParams.update({"font.size": 11, "font.family": "serif"})


def w_to_delta(positions: np.ndarray, weights: np.ndarray, d: float) -> float:
    """W(delta_d, P) for a 1D distribution: for a degenerate target this is
    exactly the weighted mean absolute distance from d (closed form of the
    1-Wasserstein/earth-mover distance to a point mass)."""
    return float(np.sum(weights * np.abs(positions - d)))


def dist_from_dict(prob_dict: dict, positions: list) -> tuple:
    """Returns (weights array aligned to positions, coverage). Missing/None
    entries contribute 0 to the vector (not imputed - just excluded from the
    sum), and coverage is the true sum so the caller can gate on it."""
    weights = np.array([prob_dict.get(str(p), None) for p in positions], dtype=object)
    weights = np.array([0.0 if w is None or (isinstance(w, float) and np.isnan(w)) else float(w) for w in weights])
    coverage = float(weights.sum())
    return weights, coverage


# ============================================================================
# Load CPR baseline + V0_true (6 anchors) distributions from consolidated.csv
# ============================================================================
cpr_df = pd.read_csv(CPR_CONSOLIDATED)
cpr_df = cpr_df[cpr_df["model"].isin(MODELS)].copy()
CPR_POSITIONS = list(range(31))
p_cols = [f"p_{i}" for i in range(31)]

cpr_baseline_dist = {}  # model -> (weights, coverage)
for m in MODELS:
    row = cpr_df[(cpr_df.model == m) & (cpr_df.condition == CPR_BASELINE_CONDITION)]
    if row.empty:
        print(f"MISSING: CPR {m} baseline row — excluded from all CPR cells")
        cpr_baseline_dist[m] = None
        continue
    r = row.iloc[0]
    weights = r[p_cols].astype(float).fillna(0.0).to_numpy()
    coverage = float(np.nansum(r[p_cols].astype(float).to_numpy()))
    cpr_baseline_dist[m] = (weights, coverage)

# cell registry: (game, model, anchor, wording) -> {"weights":..., "coverage":...} or None (excluded)
cells = {}

for m in MODELS:
    for d in CPR_ANCHORS:
        row = cpr_df[(cpr_df.model == m) & (cpr_df.condition == f"default={d}")]
        if row.empty:
            print(f"MISSING: CPR {m} default={d} V0_true — no row")
            cells[("CPR", m, d, "V0_true")] = None
            continue
        r = row.iloc[0]
        weights = r[p_cols].astype(float).fillna(0.0).to_numpy()
        coverage = float(np.nansum(r[p_cols].astype(float).to_numpy()))
        cells[("CPR", m, d, "V0_true")] = {"weights": weights, "coverage": coverage}

# ============================================================================
# Load CPR wording-robustness distributions (anchors 18/26, V1-V4) from JSONs
# ============================================================================
for path in sorted(glob.glob(os.path.join(CPR_ROBUSTNESS_DIR, "*.json"))):
    fname = os.path.basename(path)
    with open(path) as f:
        d = json.load(f)
    model = d.get("model")
    if model not in MODELS:
        continue
    anchor = d.get("default")
    variant = d.get("variant")
    if anchor not in CPR_ROBUSTNESS_ANCHORS or variant not in CPR_ROBUSTNESS_WORDINGS:
        continue
    key = ("CPR", model, anchor, variant)
    if "error" in d:
        print(f"MISSING: CPR {model} default={anchor} {variant} — FAILED cell ({d['error'][:60]}...)")
        cells[key] = None
        continue
    probs = d["cpr"][0]["sample-0"]
    weights, coverage = dist_from_dict(probs, CPR_POSITIONS)
    cells[key] = {"weights": weights, "coverage": coverage}

# ============================================================================
# Load CRD baseline + all (anchor, wording) distributions from JSONs
# ============================================================================
CRD_POSITIONS = [0, 2, 4]
crd_baseline_dist = {}
for m in MODELS:
    path = os.path.join(CRD_OUTPUT_DIR, f"{m}-crd-baseline-v0-n1-blinded.json")
    if not os.path.exists(path):
        print(f"MISSING: CRD {m} baseline JSON — excluded from all CRD cells")
        crd_baseline_dist[m] = None
        continue
    with open(path) as f:
        d = json.load(f)
    probs = d["crd"][0]["sample-0"]
    weights, coverage = dist_from_dict(probs, CRD_POSITIONS)
    crd_baseline_dist[m] = (weights, coverage)

for path in sorted(glob.glob(os.path.join(CRD_OUTPUT_DIR, "*.json"))):
    fname = os.path.basename(path)
    if "-baseline-" in fname:
        continue
    with open(path) as f:
        d = json.load(f)
    model = d.get("model")
    if model not in MODELS:
        continue
    condition = d.get("condition", "")
    if not str(condition).startswith("default="):
        continue
    anchor = int(condition.split("=")[1])
    variant = d.get("variant")
    if anchor not in CRD_ANCHORS or variant not in CRD_WORDINGS:
        continue
    key = ("CRD", model, anchor, variant)
    if "error" in d:
        print(f"MISSING: CRD {model} default={anchor} {variant} — FAILED cell")
        cells[key] = None
        continue
    probs = d["crd"][0]["sample-0"]
    weights, coverage = dist_from_dict(probs, CRD_POSITIONS)
    cells[key] = {"weights": weights, "coverage": coverage}

# ============================================================================
# Compute net_pull, W_D_base, W_D_def, w_pull for every cell
# ============================================================================
records = []
excluded_low_coverage = []
for (game, model, anchor, wording), cell in cells.items():
    positions = np.array(CPR_POSITIONS if game == "CPR" else CRD_POSITIONS, dtype=float)
    base = cpr_baseline_dist[model] if game == "CPR" else crd_baseline_dist.get(model)

    if cell is None:
        records.append({"game": game, "model": model, "anchor": anchor, "wording": wording,
                        "raw_spike": np.nan, "base_p": np.nan, "net_pull": np.nan,
                        "W_D_base": np.nan, "W_D_def": np.nan, "w_pull": np.nan,
                        "coverage_def": np.nan, "coverage_base": np.nan,
                        "excluded": True, "exclude_reason": "failed_call_or_missing"})
        continue
    if base is None:
        records.append({"game": game, "model": model, "anchor": anchor, "wording": wording,
                        "raw_spike": np.nan, "base_p": np.nan, "net_pull": np.nan,
                        "W_D_base": np.nan, "W_D_def": np.nan, "w_pull": np.nan,
                        "coverage_def": cell["coverage"], "coverage_base": np.nan,
                        "excluded": True, "exclude_reason": "missing_baseline"})
        continue

    base_weights, base_coverage = base
    def_weights, def_coverage = cell["weights"], cell["coverage"]

    if def_coverage < COVERAGE_MIN or base_coverage < COVERAGE_MIN:
        excluded_low_coverage.append((game, model, anchor, wording, def_coverage, base_coverage))
        records.append({"game": game, "model": model, "anchor": anchor, "wording": wording,
                        "raw_spike": np.nan, "base_p": np.nan, "net_pull": np.nan,
                        "W_D_base": np.nan, "W_D_def": np.nan, "w_pull": np.nan,
                        "coverage_def": def_coverage, "coverage_base": base_coverage,
                        "excluded": True, "exclude_reason": "coverage<0.90"})
        continue

    idx = int(np.where(positions == anchor)[0][0])
    raw_spike = def_weights[idx]
    base_p = base_weights[idx]
    net_pull = raw_spike - base_p

    W_D_base = w_to_delta(positions, base_weights, anchor)
    W_D_def = w_to_delta(positions, def_weights, anchor)
    # tolerance, not exact equality: floating-point residue from a baseline
    # that's degenerate-but-not-exactly-1.0 at the anchor (e.g. p=0.999999998)
    # leaves W_D_base ~1e-9, not 0.0, which would otherwise blow the ratio up
    # to nonsense (-46 was observed for Llama3.3 CRD d=4 before this fix).
    if W_D_base < 1e-6:
        w_pull = np.nan
        exclude_reason = f"W_D_base~0 ({W_D_base:.2e}, baseline already degenerate at default)"
        excluded = True
    else:
        w_pull = 1 - (W_D_def / W_D_base)
        exclude_reason = ""
        excluded = False

    records.append({"game": game, "model": model, "anchor": anchor, "wording": wording,
                    "raw_spike": raw_spike, "base_p": base_p, "net_pull": net_pull,
                    "W_D_base": W_D_base, "W_D_def": W_D_def, "w_pull": w_pull,
                    "coverage_def": def_coverage, "coverage_base": base_coverage,
                    "excluded": excluded, "exclude_reason": exclude_reason})

derived = pd.DataFrame.from_records(records)
derived_path = os.path.join(OUT_DIR, "derived_metrics_twogame_v2.csv")
derived.to_csv(derived_path, index=False)
print(f"\nWrote {derived_path} ({len(derived)} rows, {derived.excluded.sum()} excluded)\n")

if excluded_low_coverage:
    print("=== Cells excluded for coverage < 0.90 ===")
    for game, model, anchor, wording, dc, bc in excluded_low_coverage:
        print(f"  {game} {disp(model)} d={anchor} {wording}: coverage_def={dc:.3f} coverage_base={bc:.3f}")
    print()

# ============================================================================
# VERDICT 1: net_pull vs w_pull at conflict anchors, per game per wording
# ============================================================================
print("=" * 70)
print("VERDICT 1: mean net_pull vs mean w_pull at conflict anchors, by wording")
print("=" * 70)
for game, anchors in [("CPR", CPR_CONFLICT_ANCHORS), ("CRD", CRD_CONFLICT_ANCHORS)]:
    print(f"-- {game} (anchors {anchors}) --")
    for w in WORDINGS4:
        sub = derived[(derived.game == game) & (derived.anchor.isin(anchors)) &
                      (derived.wording == w) & (~derived.excluded)]
        np_mean = sub["net_pull"].mean()
        wp_mean = sub["w_pull"].mean()
        print(f"  {w}: mean net_pull={np_mean:.3f}  mean w_pull={wp_mean:.3f}  (n={len(sub)})")
    v0 = derived[(derived.game == game) & (derived.anchor.isin(anchors)) & (derived.wording == "V0_true") & (~derived.excluded)]
    v2 = derived[(derived.game == game) & (derived.anchor.isin(anchors)) & (derived.wording == "V2") & (~derived.excluded)]
    v3 = derived[(derived.game == game) & (derived.anchor.isin(anchors)) & (derived.wording == "V3") & (~derived.excluded)]
    holds_np = v2["net_pull"].mean() < v0["net_pull"].mean() < v3["net_pull"].mean() if not (v0.empty or v2.empty or v3.empty) else None
    holds_wp = v2["w_pull"].mean() < v0["w_pull"].mean() < v3["w_pull"].mean() if not (v0.empty or v2.empty or v3.empty) else None
    print(f"  V-shape (V2 < V0_true < V3) under net_pull: {holds_np}")
    print(f"  V-shape (V2 < V0_true < V3) under w_pull:   {holds_wp}")
    print()

# ============================================================================
# VERDICT 2: conflict-conditionality under w_pull (THE key rebuild)
# ============================================================================
print("=" * 70)
print("VERDICT 2: conflict-conditionality — does CPR become conflict-conditional under w_pull?")
print("=" * 70)

CRD_AGREEMENT_MODELS = [m for m in MODELS if m != "gpt_5.4_mini"]

cpr_v0 = derived[(derived.game == "CPR") & (derived.wording == "V0_true") & (~derived.excluded)]
cpr_baseline_choice = {}
for m in MODELS:
    msub = cpr_v0[cpr_v0.model == m].dropna(subset=["base_p"])
    cpr_baseline_choice[m] = int(msub.loc[msub["base_p"].idxmax(), "anchor"]) if not msub.empty else None

crd_baseline_choice = {}
for m in CRD_AGREEMENT_MODELS:
    bd = crd_baseline_dist.get(m)
    if bd is None:
        crd_baseline_choice[m] = None
        continue
    weights, _ = bd
    crd_baseline_choice[m] = int(CRD_POSITIONS[int(np.argmax(weights))])


def label_agreement(game, model, anchor):
    if game == "CRD" and model not in CRD_AGREEMENT_MODELS:
        return None
    choice = cpr_baseline_choice[model] if game == "CPR" else crd_baseline_choice[model]
    if choice is None:
        return None
    return "AGREEMENT" if anchor == choice else "CONFLICT"


derived["agreement"] = derived.apply(lambda r: label_agreement(r["game"], r["model"], r["anchor"]), axis=1)
agree_sub = derived[(derived.wording == "V0_true") & derived.agreement.notna() & (~derived.excluded)]

game_stats = {}
for game in ["CPR", "CRD"]:
    gsub = agree_sub[agree_sub.game == game]
    row = {}
    for metric in ["net_pull", "w_pull"]:
        st = gsub.groupby("agreement")[metric].agg(["mean", "count"])
        a_mean = st.loc["AGREEMENT", "mean"] if "AGREEMENT" in st.index else np.nan
        a_n = int(st.loc["AGREEMENT", "count"]) if "AGREEMENT" in st.index else 0
        c_mean = st.loc["CONFLICT", "mean"] if "CONFLICT" in st.index else np.nan
        c_n = int(st.loc["CONFLICT", "count"]) if "CONFLICT" in st.index else 0
        row[metric] = {"a_mean": a_mean, "a_n": a_n, "c_mean": c_mean, "c_n": c_n}
    game_stats[game] = row
    print(f"-- {game} --")
    for metric in ["net_pull", "w_pull"]:
        r = row[metric]
        print(f"  {metric}: Agreement mean={r['a_mean']:.3f} (n={r['a_n']})   Conflict mean={r['c_mean']:.3f} (n={r['c_n']})")
    print()

cpr_np = game_stats["CPR"]["net_pull"]
cpr_wp = game_stats["CPR"]["w_pull"]
print(f"CPR under net_pull: agreement={cpr_np['a_mean']:.3f}, conflict={cpr_np['c_mean']:.3f} "
     f"-> {'conflict-conditional' if cpr_np['a_mean'] < 0.10 else 'NOT conflict-conditional (agreement pull stays high)'}")
print(f"CPR under w_pull:   agreement={cpr_wp['a_mean']:.3f}, conflict={cpr_wp['c_mean']:.3f} "
     f"-> {'conflict-conditional' if cpr_wp['a_mean'] < 0.10 else 'NOT conflict-conditional (agreement pull stays high)'}")
if cpr_wp['a_mean'] < 0.10 and cpr_np['a_mean'] >= 0.10:
    print("=> CPR conflict-conditionality FLIPS from weak to supported under the distance-aware metric.")
elif cpr_wp['a_mean'] >= 0.10 and cpr_np['a_mean'] >= 0.10:
    print("=> CPR stays NOT conflict-conditional under both metrics — agreement pull is genuinely high, "
         "not an artefact of net_pull missing neighbourhood mass.")
else:
    print("=> CPR conflict-conditionality verdict is metric-dependent in a more complex way (check the numbers above).")
print()

# ============================================================================
# VERDICT 3: Spearman correlation between net_pull and w_pull, per game
# ============================================================================
print("=" * 70)
print("VERDICT 3: Spearman(net_pull, w_pull), per game, all cells")
print("=" * 70)
for game in ["CPR", "CRD"]:
    sub = derived[(derived.game == game) & (~derived.excluded)].dropna(subset=["net_pull", "w_pull"])
    if len(sub) >= 3:
        r, p = stats.spearmanr(sub["net_pull"], sub["w_pull"])
        print(f"  {game}: r={r:.3f}, p={p:.3g} (n={len(sub)})")
    else:
        print(f"  {game}: insufficient data (n={len(sub)})")
print()

# ============================================================================
# FIGURE 1: framing_effect_bothgames_wpull.pdf (2x2: net_pull top, w_pull bottom)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True)
for row_i, metric in enumerate(["net_pull", "w_pull"]):
    for col_i, (game, anchors, marker) in enumerate([("CPR", CPR_CONFLICT_ANCHORS, "o"), ("CRD", CRD_CONFLICT_ANCHORS, "s")]):
        ax = axes[row_i, col_i]
        for d in anchors:
            sub = derived[(derived.game == game) & (derived.anchor == d) & (~derived.excluded)]
            means = sub.groupby("wording")[metric].mean().reindex(WORDINGS4)
            ax.plot(range(4), means.values, marker=marker, label=f"d={d}", linewidth=1.8, markersize=5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["V0t", "V1", "V2", "V3"])
        if row_i == 0:
            ax.set_title(f"{game} ({'extract' if game == 'CPR' else 'contribute'})", fontsize=10)
        if col_i == 0:
            ax.set_ylabel("net_pull" if row_i == 0 else "w_pull")
        ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "framing_effect_bothgames_wpull.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "framing_effect_bothgames_wpull.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/framing_effect_bothgames_wpull.{pdf,png}")

# ============================================================================
# FIGURE 2: conflict_vs_agreement_wpull.pdf (net_pull vs w_pull, side by side)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
for ax, metric in zip(axes, ["net_pull", "w_pull"]):
    bar_data = []
    for game in ["CPR", "CRD"]:
        s = game_stats[game][metric]
        bar_data.append((game, "AGREEMENT", f"{game}\nAgreement\n(n={s['a_n']})", s["a_mean"], "#009E73"))
        bar_data.append((game, "CONFLICT", f"{game}\nConflict\n(n={s['c_n']})", s["c_mean"], "#D55E00"))
    xs = np.arange(len(bar_data))
    vals = [b[3] for b in bar_data]
    colors = [b[4] for b in bar_data]
    ax.bar(xs, vals, color=colors, alpha=0.35, edgecolor="black", linewidth=0.5, width=0.65, zorder=1)

    rng = np.random.default_rng(0)
    plotted_models = set()
    for i, (game, label, xticklbl, v, c) in enumerate(bar_data):
        gsub = agree_sub[(agree_sub.game == game) & (agree_sub.agreement == label)]
        for _, row in gsub.iterrows():
            jitter = rng.uniform(-0.18, 0.18)
            ax.scatter(i + jitter, row[metric], color=MODEL_COLORS[row["model"]],
                      edgecolor="black", linewidth=0.4, s=28, zorder=3,
                      label=disp(row["model"]) if row["model"] not in plotted_models else None)
            plotted_models.add(row["model"])
        if pd.notna(v):
            ax.text(i, v + (0.05 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=8, fontweight="bold", zorder=4)
    ax.axhline(0, color="gray", linewidth=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[2] for b in bar_data], fontsize=7.5)
    ax.set_ylabel(metric)
handles, labels = axes[0].get_legend_handles_labels()
disp_order = [disp(m) for m in MODELS]
order = [disp_order.index(l) if l in disp_order else 99 for l in labels]
paired_idx = sorted(range(len(order)), key=lambda i: order[i])
handles = [handles[i] for i in paired_idx]
labels = [labels[i] for i in paired_idx]
fig.legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), title="model", title_fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "conflict_vs_agreement_wpull.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(FIGURES_DIR, "conflict_vs_agreement_wpull.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote figures/conflict_vs_agreement_wpull.{pdf,png}")

# ============================================================================
# TABLE: framing_permodel_wpull.tex
# ============================================================================
framing_matrix = {}
for game, anchors in [("CPR", CPR_CONFLICT_ANCHORS), ("CRD", CRD_CONFLICT_ANCHORS)]:
    for m in MODELS:
        for w in WORDINGS4:
            sub = derived[(derived.game == game) & (derived.model == m) & (derived.anchor.isin(anchors)) &
                          (derived.wording == w) & (~derived.excluded)]
            framing_matrix[(game, m, w, "net_pull")] = sub["net_pull"].mean() if not sub.empty else np.nan
            framing_matrix[(game, m, w, "w_pull")] = sub["w_pull"].mean() if not sub.empty else np.nan

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    for metric in ["net\\_pull", "w\\_pull"]:
        mkey = "net_pull" if metric == "net\\_pull" else "w_pull"
        cpr_vals = " & ".join(f"{framing_matrix[('CPR', m, w, mkey)]:+.2f}"
                             if pd.notna(framing_matrix[('CPR', m, w, mkey)]) else "--" for w in WORDINGS4)
        crd_vals = " & ".join(f"{framing_matrix[('CRD', m, w, mkey)]:+.2f}"
                             if pd.notna(framing_matrix[('CRD', m, w, mkey)]) else "--" for w in WORDINGS4)
        rows_tex.append(f"{label} ({metric}) & {cpr_vals} & {crd_vals} \\\\")

framing_tex = r"""\begin{table}[t]
\centering
\caption{Per-model framing effect under both metrics: mean net\_pull and w\_pull at the conflict anchors (CPR: 18, 26; CRD: 0, 2), by wording. -- = excluded (failed call, missing baseline, or coverage $<0.90$).}
\label{tab:framing_permodel_wpull}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{CPR} & \multicolumn{4}{c}{CRD} \\
Model (metric) & V0t & V1 & V2 & V3 & V0t & V1 & V2 & V3 \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
framing_path = os.path.join(TABLES_DIR, "framing_permodel_wpull.tex")
with open(framing_path, "w") as f:
    f.write(framing_tex)
print(f"Wrote {framing_path}")

# ============================================================================
# TABLE: conflict_agreement_permodel_wpull.tex
# ============================================================================
rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    cpr_choice = cpr_baseline_choice[m]
    cpr_agree = derived[(derived.game == "CPR") & (derived.model == m) & (derived.wording == "V0_true") &
                        (derived.anchor == cpr_choice) & (~derived.excluded)]
    cpr_conflict = derived[(derived.game == "CPR") & (derived.model == m) & (derived.wording == "V0_true") &
                           (derived.anchor != cpr_choice) & (~derived.excluded)]
    cpr_a_np = cpr_agree["net_pull"].mean() if not cpr_agree.empty else np.nan
    cpr_a_wp = cpr_agree["w_pull"].mean() if not cpr_agree.empty else np.nan
    cpr_c_np = cpr_conflict["net_pull"].mean() if not cpr_conflict.empty else np.nan
    cpr_c_wp = cpr_conflict["w_pull"].mean() if not cpr_conflict.empty else np.nan

    if m in CRD_AGREEMENT_MODELS:
        crd_choice = crd_baseline_choice[m]
        crd_agree = derived[(derived.game == "CRD") & (derived.model == m) & (derived.wording == "V0_true") &
                            (derived.anchor == crd_choice) & (~derived.excluded)]
        crd_conflict = derived[(derived.game == "CRD") & (derived.model == m) & (derived.wording == "V0_true") &
                               (derived.anchor != crd_choice) & (~derived.excluded)]
        crd_a_np = crd_agree["net_pull"].mean() if not crd_agree.empty else np.nan
        crd_a_wp = crd_agree["w_pull"].mean() if not crd_agree.empty else np.nan
        crd_c_np = crd_conflict["net_pull"].mean() if not crd_conflict.empty else np.nan
        crd_c_wp = crd_conflict["w_pull"].mean() if not crd_conflict.empty else np.nan
        crd_str = (f"{crd_a_np:+.2f}/{crd_a_wp:+.2f} & {crd_c_np:+.2f}/{crd_c_wp:+.2f}"
                  if pd.notna(crd_a_np) else "-- & --")
    else:
        crd_str = "excluded & excluded"

    rows_tex.append(f"{label} & {cpr_a_np:+.2f}/{cpr_a_wp:+.2f} & {cpr_c_np:+.2f}/{cpr_c_wp:+.2f} & {crd_str} \\\\")

conflict_tex = r"""\begin{table}[t]
\centering
\caption{Per-model conflict vs.\ agreement (V0\_true), both metrics reported as net\_pull/w\_pull. Preferred anchor/choice = highest baseline probability (CPR) or baseline argmax (CRD).}
\label{tab:conflict_agreement_permodel_wpull}
\small
\begin{tabular}{lllll}
\toprule
 & \multicolumn{2}{c}{CPR} & \multicolumn{2}{c}{CRD} \\
Model & agree (np/wp) & conflict (np/wp) & agree (np/wp) & conflict (np/wp) \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
conflict_path = os.path.join(TABLES_DIR, "conflict_agreement_permodel_wpull.tex")
with open(conflict_path, "w") as f:
    f.write(conflict_tex)
print(f"Wrote {conflict_path}")

# ============================================================================
# TABLE: comprehension_wpull.tex
# ============================================================================
cpr_comp = pd.read_csv(CPR_COMPREHENSION_CSV)
crd_comp = pd.read_csv(CRD_COMPREHENSION_CSV) if os.path.exists(CRD_COMPREHENSION_CSV) else None

comp_lookup = {}
for m in MODELS:
    row = cpr_comp[cpr_comp.model == m]
    comp_lookup[m] = row.iloc[0]["mean_fraction_correct"] if not row.empty else np.nan

fixation_by_game_metric = {}
for game, anchors in [("CPR", CPR_CONFLICT_ANCHORS), ("CRD", CRD_CONFLICT_ANCHORS)]:
    for metric in ["net_pull", "w_pull"]:
        d_ = {}
        for m in MODELS:
            sub = derived[(derived.game == game) & (derived.model == m) & (derived.anchor.isin(anchors)) &
                          (derived.wording == "V0_true") & (~derived.excluded)]
            d_[m] = sub[metric].mean() if not sub.empty else np.nan
        fixation_by_game_metric[(game, metric)] = d_

rows_tex = []
for m in MODELS:
    label = disp_tex(m)
    c = comp_lookup[m]
    c_str = f"{c:.2f}" if pd.notna(c) else "--"
    vals = []
    for game in ["CPR", "CRD"]:
        for metric in ["net_pull", "w_pull"]:
            v = fixation_by_game_metric[(game, metric)][m]
            vals.append(f"{v:+.2f}" if pd.notna(v) else "--")
    rows_tex.append(f"{label} & {c_str} & " + " & ".join(vals) + r" \\")

comprehension_tex = r"""\begin{table}[t]
\centering
\caption{Comprehension score vs.\ default-following under both metrics (mean at conflict anchors, V0\_true).}
\label{tab:comprehension_wpull}
\small
\begin{tabular}{lrrrrr}
\toprule
Model & Comprehension & CPR net\_pull & CPR w\_pull & CRD net\_pull & CRD w\_pull \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
comprehension_path = os.path.join(TABLES_DIR, "comprehension_wpull.tex")
with open(comprehension_path, "w") as f:
    f.write(comprehension_tex)
print(f"Wrote {comprehension_path}")

for game in ["CPR", "CRD"]:
    for metric in ["net_pull", "w_pull"]:
        xs, ys = [], []
        for m in MODELS:
            c = comp_lookup[m]
            fx = fixation_by_game_metric[(game, metric)][m]
            if pd.notna(c) and pd.notna(fx):
                xs.append(c)
                ys.append(fx)
        if len(xs) >= 3:
            r, p = stats.spearmanr(xs, ys)
            print(f"Comprehension vs {metric}, {game}: Spearman r={r:.3f}, p={p:.3f} (n={len(xs)})")

print("\n=== FILES WRITTEN ===")
for p in [derived_path, framing_path, conflict_path, comprehension_path,
         os.path.join(FIGURES_DIR, "framing_effect_bothgames_wpull.pdf"),
         os.path.join(FIGURES_DIR, "framing_effect_bothgames_wpull.png"),
         os.path.join(FIGURES_DIR, "conflict_vs_agreement_wpull.pdf"),
         os.path.join(FIGURES_DIR, "conflict_vs_agreement_wpull.png")]:
    print(p)
