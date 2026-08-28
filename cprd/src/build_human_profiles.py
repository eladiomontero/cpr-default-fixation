"""
Build human behavior profile CSVs from the raw experiment data (complete.csv).

Output files mirror the structure of human_svo_profiles_beta.csv and are
written to cprd/data/.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIMENT_FILE = Path(__file__).parent.parent / "experiment" / "complete.csv"


def build_svo_profiles() -> pd.DataFrame:
    """
    Convert raw SVO responses into per-profile counts.

    Columns: 1,2,3,4,5,6,svo,count,role
    where 1-6 are the option index (0-8) chosen in each question and
    svo is the svo_score from the experiment data.
    """
    df = pd.read_csv(EXPERIMENT_FILE)
    svo_values = pd.read_csv(DATA_DIR / "svo_values.csv")

    # Build lookup: (question_idx, self_payoff, other_payoff) -> option value (0-8)
    lookup = {
        (int(row.question_idx), int(row.self), int(row.other)): int(row.value)
        for _, row in svo_values.iterrows()
    }

    you_cols = [f"svo.1.player.you{i}" for i in range(1, 7)]
    other_cols = [f"svo.1.player.other{i}" for i in range(1, 7)]
    needed = you_cols + other_cols + ["svo_score", "svo.1.player.role_svo"]

    sub = df[needed].dropna().copy()

    # Map each (question, self, other) to the 0-8 option index
    options = {}
    for q in range(1, 7):
        you_col = f"svo.1.player.you{q}"
        other_col = f"svo.1.player.other{q}"
        options[q] = sub.apply(
            lambda r, q=q, yc=you_col, oc=other_col: lookup.get(
                (q, int(round(r[yc])), int(round(r[oc]))), None
            ),
            axis=1,
        )

    profile = pd.DataFrame({str(q): options[q] for q in range(1, 7)})
    profile["svo"] = sub["svo_score"].values
    profile["role"] = sub["svo.1.player.role_svo"].values

    # Drop rows where any question mapping failed
    profile = profile.dropna()
    for q in range(1, 7):
        profile[str(q)] = profile[str(q)].astype(int)

    result = (
        profile.groupby([str(q) for q in range(1, 7)] + ["svo", "role"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    out_path = DATA_DIR / "human_svo_profiles.csv"
    result.to_csv(out_path, index=False)
    print(f"SVO profiles written to {out_path} ({len(result)} unique profiles)")
    return result


def build_risk_profiles() -> pd.DataFrame:
    """
    Build frequency counts of gamble choices from the risk assessment task.

    Columns: gamble_choice, count
    gamble_choice is an integer 0-5 indicating which gamble was selected.
    """
    df = pd.read_csv(EXPERIMENT_FILE)

    sub = df[["risk_assessment.1.player.gamble_choice"]].dropna().copy()
    sub["gamble_choice"] = sub["risk_assessment.1.player.gamble_choice"].astype(int)

    result = (
        sub.groupby("gamble_choice")
        .size()
        .reset_index(name="count")
        .sort_values("gamble_choice")
        .reset_index(drop=True)
    )

    out_path = DATA_DIR / "human_risk_profiles.csv"
    result.to_csv(out_path, index=False)
    print(f"Risk profiles written to {out_path} ({len(result)} rows)")
    return result


def build_cpr_profiles() -> pd.DataFrame:
    """
    Build frequency counts of extraction sequences across CPR rounds.

    Columns: extraction1,...,extraction10,Treatment,count
    Each row is a unique (extraction sequence, Treatment) combination
    with the number of participants who played that exact sequence.
    """
    df = pd.read_csv(EXPERIMENT_FILE)

    extraction_cols = [f"extraction{i}" for i in range(1, 11)]
    needed = extraction_cols + ["Treatment"]
    sub = df[needed].dropna().copy()

    for col in extraction_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()

    result = (
        sub.groupby(extraction_cols + ["Treatment"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    out_path = DATA_DIR / "human_cpr_profiles.csv"
    result.to_csv(out_path, index=False)
    print(f"CPR profiles written to {out_path} ({len(result)} unique profiles)")
    return result


if __name__ == "__main__":
    build_svo_profiles()
    build_risk_profiles()
    build_cpr_profiles()
