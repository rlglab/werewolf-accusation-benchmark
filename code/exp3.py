"""Experiment 3 — belief shifts by prior trust in the accuser (paper Table 2).

Observations are binned by the model's own stage-1 prior belief about the
accuser: Trust {-3,-2}, Neutral {-1,0,+1}, Distrust {+2,+3}. Cells are the
mean stage-2 belief shift of the target and of the accuser, split by the
accuser's true team, per model-size group.

    python exp3.py --results ../results --out-dir ../outputs

Writes exp3_shift_by_trust.csv (size groups), exp3_per_model.csv and
exp3_size_trend.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ci95, common_args, fmt, group_rows, load, mean_or_nan, per_model_rows, trend_table

BINS = ("trust", "neutral", "distrust")
CELLS = [(team, party, b) for team in ("villager", "wolf") for party in ("target", "accuser") for b in BINS]


def trust_bin(prior: int) -> str:
    if prior <= -2:
        return "trust"
    if prior >= 2:
        return "distrust"
    return "neutral"


def add_trust_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast each (model, spec)'s prior belief about the accuser to all its rows."""
    acc = (df[df.rated_party == "accuser"][["model_id", "spec_id", "prior_belief"]]
           .drop_duplicates(["model_id", "spec_id"])
           .rename(columns={"prior_belief": "accuser_prior"}))
    df = df.merge(acc, on=["model_id", "spec_id"], how="inner")
    df["trust_bin"] = df.accuser_prior.map(trust_bin)
    return df


def col(team: str, party: str, b: str) -> str:
    return f"{team}_accuser__{party}_shift__{b}"


def cells(sub: pd.DataFrame) -> dict:
    out = {}
    for team, party, b in CELLS:
        s = sub.loc[(sub.accuser_team == team) & (sub.rated_party == party)
                    & (sub.trust_bin == b), "belief_shift"]
        out[col(team, party, b)] = mean_or_nan(s)
        out[col(team, party, b) + "_ci95"] = ci95(s)
        out[col(team, party, b) + "_n"] = len(s)
    return out


def main() -> None:
    args = common_args(__doc__).parse_args()
    df = add_trust_bin(load(args))
    out = Path(args.out_dir)

    table = group_rows(df, cells)
    table.to_csv(out / "exp3_shift_by_trust.csv", index=False)
    per_model = per_model_rows(df, cells)
    per_model.to_csv(out / "exp3_per_model.csv", index=False)
    trend = trend_table(per_model, [col(*c) for c in CELLS])
    trend.to_csv(out / "exp3_size_trend.csv", index=False)

    for team in ("villager", "wolf"):
        print(f"\n== Exp 3: {team} as accuser (target/accuser shift by trust bin) ==")
        show = table[["size_group", "n_models"] + [col(team, p, b) for p in ("target", "accuser") for b in BINS]]
        show = show.rename(columns=lambda c: c.replace(f"{team}_accuser__", "").replace("_shift__", "/"))
        print(fmt(show))
    print("\n== Spearman (per-model mean vs. size_b) ==")
    print(trend.to_string(index=False))
    print(f"\nwrote {out}/exp3_*.csv")


if __name__ == "__main__":
    main()
