import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import streamlit as st

from utilsOutput import CATEGORICAL_COLORS, LINESTYLES, load_run_row, parse_run_filename

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data
def scan_runs(input_dir: str):
    """Parse every output JSON under input_dir into a flat list of run records."""
    runs = []
    for path in sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)):
        info = parse_run_filename(os.path.basename(path))
        if info is None:
            continue
        runs.append({"path": path, **info})
    return runs


def parse_default_input_dir() -> str:
    """Read --input_dir from argv (passed after `--` by `streamlit run`), if given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.input_dir


def main():
    st.set_page_config(page_title="CPRD results viewer", layout="wide")
    st.title("CPRD results viewer")

    input_dir = st.sidebar.text_input("Input folder", value=parse_default_input_dir())
    input_dir_abs = input_dir if os.path.isabs(input_dir) else os.path.join(PROJECT_DIR, input_dir)

    runs = scan_runs(input_dir_abs)
    if not runs:
        st.warning(f"No matching output JSON files found under {input_dir_abs}")
        return

    tasks = sorted({r["task"] for r in runs})
    task = st.sidebar.selectbox("Task", tasks, index=tasks.index("cpr") if "cpr" in tasks else 0)
    runs = [r for r in runs if r["task"] == task]

    models = sorted({r["model_id"] for r in runs})
    conditions = sorted({r["condition"] for r in runs})
    versions = sorted({r["version"] for r in runs})
    system_types = sorted({r["system_type"] for r in runs})
    nsamples = sorted({r["nsample"] for r in runs}, key=int)

    sel_models = st.sidebar.multiselect("Model(s)", models, default=models)
    sel_conditions = st.sidebar.multiselect("Condition(s)", conditions, default=conditions)
    sel_versions = st.sidebar.multiselect("Version(s)", versions, default=versions)
    sel_system_types = st.sidebar.multiselect("System type(s)", system_types, default=system_types)
    sel_nsamples = st.sidebar.multiselect("nsample(s)", nsamples, default=nsamples)

    color_by = st.sidebar.radio("Color by", ["model", "condition"], index=0)

    filtered = [
        r for r in runs
        if r["model_id"] in sel_models
        and r["condition"] in sel_conditions
        and r["version"] in sel_versions
        and r["system_type"] in sel_system_types
        and r["nsample"] in sel_nsamples
    ]

    st.caption(f"{len(filtered)} run(s) match the current filters (out of {len(runs)} total for task '{task}').")

    if not filtered:
        st.info("No runs match the current filter selection.")
        return

    key_field = "model_id" if color_by == "model" else "condition"
    key_values = sorted({r[key_field] for r in filtered})
    n_colors = len(CATEGORICAL_COLORS)
    color_map = {
        key: (CATEGORICAL_COLORS[i % n_colors], LINESTYLES[(i // n_colors) % len(LINESTYLES)])
        for i, key in enumerate(key_values)
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    choices = None
    skipped = []
    for r in filtered:
        run_choices, averages, notes = load_run_row(r["path"], task)
        if not run_choices:
            skipped.append(f"{os.path.basename(r['path'])} ({notes})")
            continue
        choices = run_choices
        key = r[key_field]
        color, linestyle = color_map[key]
        label = f"{r['model_id']} | {r['condition']} | v{r['version']} n{r['nsample']} {r['system_type']}"
        ax.plot(range(len(run_choices)), averages, marker="o", markersize=6,
               markeredgecolor="#fcfcfb", markeredgewidth=1, linestyle=linestyle,
               linewidth=1.8, label=label, color=color, zorder=3)

    if skipped:
        st.warning(f"Skipped {len(skipped)} run(s): " + "; ".join(skipped))

    if choices is None:
        st.info("None of the matching runs had valid samples to plot.")
        return

    positions = list(range(len(choices)))
    tick_step = 5 if len(choices) > 10 else 1
    ax.set_xticks(positions[::tick_step], choices[::tick_step])
    ax.set_ylim(0, 1)
    ax.set_title(f"Average extraction probability ({task})", color="#0b0b0b")
    ax.set_xlabel("choice", color="#52514e")
    ax.set_ylabel("average probability", color="#52514e")
    ax.tick_params(colors="#898781")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper right", fontsize=8, labelcolor="#0b0b0b")
    fig.tight_layout()

    st.pyplot(fig)

    with st.expander("Matching files"):
        st.table([{k: r[k] for k in ("model_id", "condition", "version", "nsample", "system_type")} for r in filtered])


if __name__ == "__main__":
    main()
