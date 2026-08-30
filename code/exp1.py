"""Experiment 1 — prior belief discrimination (paper Fig. 2 / Table 4).

Mean stage-1 prior belief (-3..+3, + = wolf) assigned to true villagers and
true wolves, per model-size group, plus the separation (wolf - villager).

    python exp1.py --results ../results --out-dir ../outputs

Writes exp1_prior_belief.csv (size groups), exp1_per_model.csv and
exp1_size_trend.csv (Spearman of the separation vs. model size).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ci95, common_args, fmt, group_rows, load, mean_or_nan, per_model_rows, trend_table


def cells(sub: pd.DataFrame) -> dict:
    v = sub.loc[sub.subject_team == "villager", "prior_belief"]
    w = sub.loc[sub.subject_team == "wolf", "prior_belief"]
    return {
        "true_villager": mean_or_nan(v), "true_villager_ci95": ci95(v), "n_villager": len(v),
        "true_wolf": mean_or_nan(w), "true_wolf_ci95": ci95(w), "n_wolf": len(w),
        "separation": mean_or_nan(w) - mean_or_nan(v),
    }


def main() -> None:
    args = common_args(__doc__).parse_args()
    df = load(args)
    out = Path(args.out_dir)

    table = group_rows(df, cells)
    # In the table, separation is reported as the difference between the two
    # rounded means shown (e.g. 0.28 - (-0.52) = 0.80), as in the paper.
    table["separation"] = table["true_wolf"].round(2) - table["true_villager"].round(2)
    table.to_csv(out / "exp1_prior_belief.csv", index=False)
    per_model = per_model_rows(df, cells)
    per_model.to_csv(out / "exp1_per_model.csv", index=False)
    trend = trend_table(per_model, ["true_villager", "true_wolf", "separation"])
    trend.to_csv(out / "exp1_size_trend.csv", index=False)

    print("\n== Exp 1: prior belief by size group ==")
    print(fmt(table[["size_group", "n_models", "true_villager", "true_wolf", "separation"]]))
    print("\n== Spearman (per-model mean vs. size_b) ==")
    print(trend.to_string(index=False))
    print(f"\nwrote {out}/exp1_*.csv")


if __name__ == "__main__":
    main()
