"""
criteria.py
Executable calibration and validation criteria derived from ASME V&V40
(Wang et al., eLife, 2026). Scores a population of simulations against biomarker
criteria and ranks them under one or more importance profiles.
"""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Criterion:
    name: str                    # unique id, e.g. "lvef"
    biomarker: str               # column name in the biomarker table
    lower: float = -np.inf
    upper: float = np.inf
    scale: float | None = None   # normalisation for the violation distance
    stage: str = "calibration"   # "calibration" or "validation"
    units: str = ""
    source: str = ""

    def _scale(self) -> float:
        if self.scale is not None:
            return self.scale
        width = self.upper - self.lower
        if np.isfinite(width) and width > 0:
            return width
        bound = self.lower if np.isfinite(self.lower) else self.upper
        return max(abs(bound), 1.0)

    def violation(self, x) -> np.ndarray:
        """0 inside [lower, upper]; normalised distance outside; NaN -> inf."""
        x = np.asarray(x, dtype=float)
        v = (np.clip(self.lower - x, 0, None) + np.clip(x - self.upper, 0, None)) / self._scale()
        return np.where(np.isnan(x), np.inf, v)


def load_criteria(path: str):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    criteria = [Criterion(name=k, **v) for k, v in cfg["criteria"].items()]
    profiles = cfg.get("profiles", {"equal": {}})
    return criteria, profiles


def violations(biomarkers: pd.DataFrame, criteria, stage: str | None = None) -> pd.DataFrame:
    if stage is not None:
        criteria = [c for c in criteria if c.stage == stage]
    missing = [c.biomarker for c in criteria if c.biomarker not in biomarkers.columns]
    if missing:
        raise KeyError(f"Biomarkers missing from input: {missing}")
    return pd.DataFrame(
        {c.name: c.violation(biomarkers[c.biomarker].to_numpy()) for c in criteria},
        index=biomarkers.index,
    )


def profile_score(V: pd.DataFrame, profile: dict):
    """Returns (score, rank). Lower score is better; rank 1 is best."""
    if "tiers" in profile:
        # Strict priority: tier 1 total violation first, later tiers break ties.
        keys = [V[list(t)].sum(axis=1).to_numpy() for t in profile["tiers"]]
        order = np.lexsort(keys[::-1])
        rank = np.empty(len(V), dtype=int)
        rank[order] = np.arange(1, len(V) + 1)
        return pd.Series(keys[0], index=V.index), pd.Series(rank, index=V.index)
    # Weighted: unlisted criteria get default_weight (1.0 unless set).
    w = (pd.Series(profile.get("weights", {}), dtype=float)
         .reindex(V.columns)
         .fillna(profile.get("default_weight", 1.0)))
    active = w[w > 0].index
    score = (V[active] * w[active]).sum(axis=1)
    return score, score.rank(method="min").astype(int)


def evaluate(biomarkers, criteria, profiles, params: pd.DataFrame | None = None,
             biomarker_names: list[str] | None = None,
             stage: str | None = None, top_k: int = 10):
    """
    biomarkers: DataFrame (samples x biomarkers), or ndarray with biomarker_names.
    params:     optional LHS parameter table with the same index, joined to output.
    Returns (summary sorted by robustness across profiles, violation matrix).
    """
    if isinstance(biomarkers, np.ndarray):
        if biomarker_names is None:
            raise ValueError("biomarker_names required for array input")
        biomarkers = pd.DataFrame(biomarkers, columns=biomarker_names)
    V = violations(biomarkers, criteria, stage)

    out = pd.DataFrame(index=V.index)
    out["n_pass"] = (V == 0).sum(axis=1)
    out["n_criteria"] = V.shape[1]
    for name, prof in profiles.items():
        out[f"score_{name}"], out[f"rank_{name}"] = profile_score(V, prof)

    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out["top_k_count"] = (out[rank_cols] <= top_k).sum(axis=1)  # profiles where sample is in top k
    out["worst_rank"] = out[rank_cols].max(axis=1)               # robustness across profiles
    if params is not None:
        out = params.join(out)
    return out.sort_values(["top_k_count", "worst_rank"], ascending=[False, True]), V