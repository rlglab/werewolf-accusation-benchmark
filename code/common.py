"""Shared code for the exp*.py scripts.

* ``load_results``  — read every ``results/<model>/`` directory into one
  observation-level table (one row per model × item × rated player with
  both a stage-1 prior and a stage-2 shift; accuser / target rows only).
* ``group_rows`` / ``per_model_rows`` / ``trend_table`` — the size-group,
  per-model and Spearman-trend tables every experiment writes.

Observation columns
-------------------
model_id, family, size_b, reasoning, size_group
spec_id, annotation_id, subject_player_name
rated_party        'accuser' | 'target'
accuser_team       true team of the accuser  ('villager' | 'wolf')
subject_team       true team of the rated player ('villager' | 'wolf')
strength           'suspect' | 'accuse'
prior_belief       -3..+3 (stage 1; + = wolf-leaning)
belief_shift       -3..+3 (stage 2; + = shifted toward wolf)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

from models import BY_SLUG, SIZE_GROUP_LABELS, size_group

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
DATA = RELEASE / "data"

# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------

TEAM_SHORT = {"Villagers": "villager", "Werewolves": "wolf"}


def _model_meta(model_dir: Path) -> dict | None:
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        m.setdefault("size_group", size_group(float(m["size_b"])))
        return m
    info = BY_SLUG.get(model_dir.name)
    if info is None:
        return None
    return {"model_id": info.model_id, "family": info.family, "size_b": info.size_b,
            "reasoning": info.reasoning, "size_group": info.size_group}


def load_one(model_dir: Path, specs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame | None:
    p1, p2 = model_dir / "part1_prior.parquet", model_dir / "part2_shift.parquet"
    meta = _model_meta(model_dir)
    if meta is None or not p1.exists() or not p2.exists():
        return None
    prior = pd.read_parquet(p1)
    shift = pd.read_parquet(p2)
    prior = prior[prior.parse_status == "ok"][["spec_id", "subject_player_name", "prior_belief"]]
    shift = shift[shift.parse_status == "ok"][["spec_id", "subject_player_name", "belief_shift"]]
    obs = prior.merge(shift, on=["spec_id", "subject_player_name"], how="inner")
    obs = obs.merge(specs, on="spec_id", how="left")
    obs = obs.merge(labels, on=["spec_id", "subject_player_name"], how="left")
    obs["rated_party"] = [
        "accuser" if name == acc else ("target" if name in list(tgts) else "other")
        for name, acc, tgts in zip(obs.subject_player_name, obs.accuser_player_name,
                                   obs.named_target_player_names)]
    obs = obs[obs.rated_party.isin(["accuser", "target"])].copy()
    for k in ("model_id", "family", "size_b", "reasoning", "size_group"):
        obs[k] = meta[k]
    obs["prior_belief"] = obs["prior_belief"].astype(int)
    obs["belief_shift"] = obs["belief_shift"].astype(int)
    return obs[[
        "model_id", "family", "size_b", "reasoning", "size_group",
        "spec_id", "annotation_id", "subject_player_name", "rated_party",
        "accuser_team", "subject_team", "strength", "prior_belief", "belief_shift",
    ]]


def load_results(results_dir: Path | str, data_dir: Path | str = DATA) -> pd.DataFrame:
    results_dir, data_dir = Path(results_dir), Path(data_dir)
    specs = pd.read_parquet(data_dir / "specs.parquet",
                            columns=["spec_id", "annotation_id", "accuser_team_truth", "strength",
                                     "accuser_player_name", "named_target_player_names"])
    specs["accuser_team"] = specs.pop("accuser_team_truth").map(TEAM_SHORT)
    labels = pd.read_parquet(data_dir / "labels.parquet")
    labels["subject_team"] = labels.pop("subject_team_truth").map(TEAM_SHORT)
    labels = labels[["spec_id", "subject_player_name", "subject_team"]]

    frames, skipped = [], []
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        obs = load_one(d, specs, labels)
        if obs is None:
            skipped.append(d.name)
        else:
            frames.append(obs)
    if skipped:
        print(f"[load_results] skipped {len(skipped)} dirs without complete results/meta: {skipped}")
    if not frames:
        raise SystemExit(f"no model results found under {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    print(f"[load_results] {df.model_id.nunique()} models, {len(df):,} observations "
          f"(size groups: {df.drop_duplicates('model_id').size_group.value_counts().to_dict()})")
    return df



# ---------------------------------------------------------------------------
# Aggregation helpers for the experiments
# ---------------------------------------------------------------------------

def common_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--results", default=str(RELEASE / "results"),
                   help="directory containing one sub-directory per model")
    p.add_argument("--out-dir", default=str(RELEASE / "outputs"), help="where the CSVs go")
    p.add_argument("--data", default=str(RELEASE / "data"))
    return p


def load(args: argparse.Namespace) -> pd.DataFrame:
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return load_results(args.results, args.data)


def group_rows(df: pd.DataFrame, cell_fn) -> pd.DataFrame:
    """One row per size group + an ``All`` row.

    ``cell_fn(sub_df) -> dict`` computes the table cells as flat means over
    all observation rows in the group (the paper's row-level averaging).
    """
    rows = []
    for g in SIZE_GROUP_LABELS:
        sub = df[df.size_group == g]
        if sub.empty:
            continue
        rows.append({"size_group": g, "n_models": sub.model_id.nunique(), **cell_fn(sub)})
    rows.append({"size_group": "All", "n_models": df.model_id.nunique(), **cell_fn(df)})
    return pd.DataFrame(rows)


def per_model_rows(df: pd.DataFrame, cell_fn) -> pd.DataFrame:
    rows = []
    for mid, sub in df.groupby("model_id", sort=False):
        first = sub.iloc[0]
        rows.append({"model_id": mid, "family": first.family, "size_b": first.size_b,
                     "reasoning": first.reasoning, "size_group": first.size_group,
                     **cell_fn(sub)})
    return pd.DataFrame(rows).sort_values(["size_b", "model_id"]).reset_index(drop=True)


def mean_or_nan(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else math.nan


def ci95(s: pd.Series) -> float:
    """95% CI half-width over observation rows (1.96 * sd / sqrt(n))."""
    if len(s) < 2:
        return math.nan
    return float(1.96 * s.std(ddof=1) / math.sqrt(len(s)))


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman rank correlation and two-sided p-value across models."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3:
        return math.nan, math.nan
    rho, p = stats.spearmanr(d.x, d.y)
    return float(rho), float(p)


def trend_table(per_model: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Spearman rho / p of each per-model column against model size."""
    rows = []
    for c in columns:
        rho, p = spearman(per_model.size_b, per_model[c])
        rows.append({"quantity": c, "n_models": int(per_model[c].notna().sum()),
                     "spearman_rho": round(rho, 4), "spearman_p": p})
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, ndigits: int = 2) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.{ndigits}f}")
