"""Latin hypercube sample of LV run configs (JSON) around a base config.

    python3 scripts/sample_lhs_lv.py configs/lv_ellipsoid/baseline.toml lv_lhs1 --n 96 --seed 20261023

Writes runs/<study>/configs/<name>.json, runs/<study>/configs.txt (for run_pool.sh /
archer2/sweep.slurm) and runs/<study>/samples.csv. Each run's output goes to runs/<study>/<name>/.
FACTORS below: dotted path into the config -> (low, high, "lin" or "log").
All factors here are mechanics or circulation parameters, so every sample shares one steady state.
"""
import argparse
import csv
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import qmc

FACTORS = {
    "land.scales.Tref": (5.0, 10.0, "lin"),
    "land.tref_sigmoid.rest_scale": (0.2, 1.0, "lin"),
    "land.tref_sigmoid.n_half": (0.05, 0.15, "lin"),
    "circulation.lv.p_end_diastole_pa": (800.0, 1500.0, "lin"),
    "circulation.lv.windkessel.p_init_pa": (8000.0, 11000.0, "lin"),
    "circulation.lv.windkessel.resistance_mmhg_s_per_ml": (0.9, 1.3, "lin"),
    "circulation.lv.windkessel.compliance_ml_per_mmhg": (0.8, 2.5, "lin"),
    "circulation.lv.windkessel.characteristic_impedance_mmhg_s_per_ml": (0.02, 0.06, "lin"),
}

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("study")
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--seed", type=int, default=20261023)
args = ap.parse_args()

base_path = Path(args.base).resolve()
with open(base_path, "rb") as f:
    base = tomllib.load(f)
base["case"] = str((base_path.parent / base["case"]).resolve())   # absolute, so configs can live anywhere
assert "tref_sigmoid" in base.get("land", {}), "base config needs a [land.tref_sigmoid] section"

study = Path("runs") / args.study
(study / "configs").mkdir(parents=True, exist_ok=True)

u = qmc.LatinHypercube(d=len(FACTORS), seed=args.seed).random(args.n)
values = np.empty_like(u)
for j, (lo, hi, scale) in enumerate(FACTORS.values()):
    values[:, j] = (np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if scale == "log"
                    else lo + u[:, j] * (hi - lo))


def set_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


rows, paths = [], []
for i in range(args.n):
    cfg = json.loads(json.dumps(base))
    for key, val in zip(FACTORS, values[i]):
        set_path(cfg, key, float(val))
    name = f"s{i:03d}"
    cfg["outdir"] = str(study / name)
    p = study / "configs" / f"{name}.json"
    p.write_text(json.dumps(cfg, indent=2))
    paths.append(str(p))
    rows.append({"name": name, **{k: float(v) for k, v in zip(FACTORS, values[i])}})

(study / "configs.txt").write_text("\n".join(paths) + "\n")
with open(study / "samples.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(study / "spec.json").write_text(json.dumps({"base": str(base_path), "n": args.n, "seed": args.seed,
                                             "factors": FACTORS}, indent=2))
print(f"{args.n} samples over {len(FACTORS)} factors in {study}; list in {study}/configs.txt")