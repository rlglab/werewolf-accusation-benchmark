"""Experiment 2 — belief shifts after accusations (paper Table 1).

Mean stage-2 belief shift (-3..+3, + = toward wolf) of the accused target and
of the accuser, split by the accuser's true team (villager / wolf) and the
message strength (suspect = soft suspicion, accuse = direct accusation),
per model-size group.

    python exp2.py --results ../results --out-dir ../outputs

Writes exp2_belief_shift.csv (size groups), exp2_per_model.csv and
exp2_size_trend.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ci95, common_args, fmt, group_rows, load, mean_or_nan, per_model_rows, trend_table

CELLS = [(team, party, strength)
         for team in ("villager", "wolf")
         for party in ("target", "accuser")
         for strength in ("suspect", "accuse")]


def col(team: str, party: str, strength: str) -> str:
    return f"{team}_accuser__{party}_shift__{strength}"


def cells(sub: pd.DataFrame) -> dict:
    out = {}
    for team, party, strength in CELLS:
        s = sub.loc[(sub.accuser_team == team) & (sub.rated_party == party)
                    & (sub.strength == strength), "belief_shift"]
        out[col(team, party, strength)] = mean_or_nan(s)
        out[col(team, party, strength) + "_ci95"] = ci95(s)
        out[col(team, party, strength) + "_n"] = len(s)
    return out


def main() -> None:
    args = common_args(__doc__).parse_args()
    df = load(args)
    out = Path(args.out_dir)

    table = group_rows(df, cells)
    table.to_csv(out / "exp2_belief_shift.csv", index=False)
    per_model = per_model_rows(df, cells)
    per_model.to_csv(out / "exp2_per_model.csv", index=False)
    trend = trend_table(per_model, [col(*c) for c in CELLS])
    trend.to_csv(out / "exp2_size_trend.csv", index=False)

    for team in ("villager", "wolf"):
        print(f"\n== Exp 2: {team} as accuser (target/accuser shift, suspect vs accuse) ==")
        show = table[["size_group", "n_models"] + [col(team, p, s) for p in ("target", "accuser")
                                                    for s in ("suspect", "accuse")]]
        show = show.rename(columns=lambda c: c.replace(f"{team}_accuser__", "").replace("_shift__", "/"))
        print(fmt(show))
    print("\n== Spearman (per-model mean vs. size_b) ==")
    print(trend.to_string(index=False))
    print(f"\nwrote {out}/exp2_*.csv")


if __name__ == "__main__":
    main()
