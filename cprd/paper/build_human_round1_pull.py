"""Human round-1 choices per CPR treatment (from the raw participant data,
cprd/experiment/complete.csv) vs. the six models' single-shot pull, same
metric (mean-shift pull ratio = (choice - control) / (default - control)),
same two anchors (11, 23). No model calls - pure post-processing.

Reads: cprd/experiment/complete.csv, cprd/paper/derived_metrics.csv
Writes: cprd/paper/tables/human_vs_model_round1_pull.tex
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cprd/
EXPERIMENT_CSV = os.path.join(ROOT, "experiment", "complete.csv")
DERIVED_CSV = os.path.join(ROOT, "paper", "derived_metrics.csv")
TABLES_DIR = os.path.join(ROOT, "paper", "tables")

TREATMENT_DEFAULT = {"Control": None, "Pro-social": 11, "Self-serving": 23}

# ----------------------------------------------------------------------------
# Human round-1 choices
# ----------------------------------------------------------------------------
human_df = pd.read_csv(EXPERIMENT_CSV, low_memory=False)
missing = human_df["extraction1"].isna().sum()
if missing:
    print(f"MISSING: {missing} participants have no extraction1 value — excluded")

human_stats = {}
for t, d in TREATMENT_DEFAULT.items():
    sub = human_df[human_df.Treatment == t]["extraction1"].dropna()
    n = len(sub)
    mean = sub.mean()
    sem = sub.sem()
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 else (np.nan, np.nan)
    human_stats[t] = {"default": d, "n": n, "mean": mean, "sd": sub.std(), "ci_lo": ci[0], "ci_hi": ci[1]}
    print(f"HUMAN {t:14s} default={d}  n={n}  round-1 mean={mean:.3f}  "
         f"95% CI=({ci[0]:.3f}, {ci[1]:.3f})")

control_mean = human_stats["Control"]["mean"]
for t, d in TREATMENT_DEFAULT.items():
    if d is None:
        continue
    s = human_stats[t]
    s["pull"] = (s["mean"] - control_mean) / (d - control_mean)
    print(f"HUMAN {t:14s} pull = ({s['mean']:.3f} - {control_mean:.3f}) / ({d} - {control_mean:.3f}) = {s['pull']:.3f}")
print()

# ----------------------------------------------------------------------------
# Model single-shot pull (mean-shift definition, matches human formula exactly)
# ----------------------------------------------------------------------------
derived = pd.read_csv(DERIVED_CSV)
model_stats = {}
for t, d in TREATMENT_DEFAULT.items():
    if d is None:
        continue
    sub = derived[derived.default == d]["pull"].dropna()
    n = len(sub)
    mean = sub.mean()
    sem = sub.sem()
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 else (np.nan, np.nan)
    model_stats[t] = {"n": n, "mean": mean, "sd": sub.std(), "ci_lo": ci[0], "ci_hi": ci[1]}
    print(f"MODELS {t:14s} default={d}  n={n}  mean pull={mean:.3f}  95% CI=({ci[0]:.3f}, {ci[1]:.3f})")
print()

# ----------------------------------------------------------------------------
# LaTeX table
# ----------------------------------------------------------------------------
rows_tex = []
for t, d in TREATMENT_DEFAULT.items():
    h = human_stats[t]
    if d is None:
        rows_tex.append(
            f"{t} (control) & {h['mean']:.2f} ({h['ci_lo']:.2f}--{h['ci_hi']:.2f}), n={h['n']} & -- & -- \\\\"
        )
        continue
    m = model_stats[t]
    rows_tex.append(
        f"{t} ($d{{=}}{d}$) & {h['mean']:.2f} ({h['ci_lo']:.2f}--{h['ci_hi']:.2f}), n={h['n']} & "
        f"{h['pull']:.2f} & {m['mean']:.2f} ({m['ci_lo']:.2f}--{m['ci_hi']:.2f}), n={m['n']} \\\\"
    )

# reorder so control's round-1 mean is shown as reference row without pull columns collapsed oddly
table_tex = r"""\begin{table}[t]
\centering
\caption{Human round-1 choices vs. model single-shot pull, CPR game. Human pull $= (\text{mean round-1 choice} - \text{control mean}) / (\text{default} - \text{control mean})$; model pull uses the identical formula on each model's single-shot mean extraction (baseline vs.\ under-default). Human control mean round-1 $= """ + f"{control_mean:.2f}" + r"""$ (n=""" + str(human_stats["Control"]["n"]) + r"""). Model column reports mean $\pm$ 95\% CI across the 6 models.}
\label{tab:human_round1_pull}
\small
\begin{tabular}{lccc}
\toprule
Treatment & Human round-1 mean (95\% CI) & Human pull & Model mean pull (95\% CI) \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
os.makedirs(TABLES_DIR, exist_ok=True)
out_path = os.path.join(TABLES_DIR, "human_vs_model_round1_pull.tex")
with open(out_path, "w") as f:
    f.write(table_tex)
print(f"Wrote {out_path}")
print()
print(table_tex)
