"""sample_ta_lhs.py: Latin hypercube sample of the prescribed-Ta parameters.

Reads a study spec (TOML) and draws one LHS over
  [factors]  multiplicative factors on the [active.endo] values of the base config
             (epi and mid follow through their fixed ratios), and
  [values]   absolute values for any config entry, by dotted path (e.g. "active.Ta_d_frac"),
then writes one JSON config per sample plus a samples table and a list file for the pool runner.

    python3 scripts/sample_ta_lhs.py configs/ta_sweep/lhs_test.toml

Output (runs/ is untracked; the spec and its seed reproduce everything):
    runs/<study>/samples.csv              sample id, factors and resulting endo values
    runs/<study>/configs.txt              one config path per line
    runs/<study>/sample_000/config.json   ...
"""
import argparse
import copy
import csv
import json
import re
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ap = argparse.ArgumentParser()
ap.add_argument("spec", help="study spec (.toml)")
args = ap.parse_args()

spec_path = Path(args.spec).resolve()
spec = tomllib.loads(spec_path.read_text())
base_path = (spec_path.parent / spec["base"]).resolve()
base = tomllib.loads(base_path.read_text())
base["case"] = str((base_path.parent / base["case"]).resolve())   # absolute: configs move to runs/

study = spec["study"]
n, seed = int(spec["n_samples"]), int(spec["seed"])
factor_ranges = spec.get("factors", {})                   # {endo key: [low, high]}
value_ranges = spec.get("values", {})                     # {dotted config path: [low, high]}
for k in factor_ranges:
    assert k in base["active"]["endo"], f"{k} is not a key of [active.endo] in {base_path.name}"
names = [f"f_{k}" for k in factor_ranges] + list(value_ranges)
bounds = list(factor_ranges.values()) + list(value_ranges.values())
lo = np.array([b[0] for b in bounds], dtype=float)
hi = np.array([b[1] for b in bounds], dtype=float)


def set_path(cfg, dotted, value):
    *parents, key = dotted.split(".")
    node = cfg
    for k in parents:
        node = node.setdefault(k, {})
    node[key] = value

unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
factors = qmc.scale(unit, lo, hi)

outroot = Path("runs") / study
outroot.mkdir(parents=True, exist_ok=True)

# optional restart from a shared preload checkpoint (Ta is zero before the first activation,
# so every sample has the same preload)
restart = ""
if spec.get("restart_dir"):
    affects_preload = [k for k in value_ranges if k.startswith(("active.Ta_d", "mechanics.", "circulation."))]
    assert not affects_preload, (f"{affects_preload} change the preload, so samples cannot share a preload "
                                 "checkpoint: remove restart_dir from the spec")
    rdir = Path(spec["restart_dir"]).resolve()
    t_max = float(spec["restart_before_ms"])
    found = []
    for f in rdir.glob("t_*.json"):
        m = re.fullmatch(r"t_([0-9.]+)\.json", f.name)
        if m and float(m.group(1)) <= t_max:
            found.append((float(m.group(1)), f))
    assert found, f"no checkpoint at or before {t_max} ms in {rdir}"
    t_restart, f = max(found)
    restart = str(f.with_suffix(""))
    print(f"all samples restart from {restart} (t = {t_restart} ms)")

rows, paths = [], []
for i, fac in enumerate(factors):
    cfg = copy.deepcopy(base)
    for k, v in spec.get("override", {}).items():          # e.g. time.t_end_ms
        set_path(cfg, k, v)
    drawn = dict(zip(names, map(float, fac)))
    for k in factor_ranges:
        cfg["active"]["endo"][k] = base["active"]["endo"][k] * drawn[f"f_{k}"]
    for k in value_ranges:
        set_path(cfg, k, drawn[k])
    cfg["outdir"] = str(outroot / f"sample_{i:03d}")
    cfg.setdefault("restart", {})["from"] = restart
    cfg["sample"] = {"study": study, "id": i, "seed": seed, "drawn": drawn}
    path = outroot / f"sample_{i:03d}" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    paths.append(str(path))
    rows.append({"id": i, **drawn, **{f"endo_{k}": cfg["active"]["endo"][k] for k in factor_ranges}})

with open(outroot / "samples.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(outroot / "configs.txt").write_text("\n".join(paths) + "\n")
print(f"{n} samples of {names} written to {outroot}/ (configs.txt, samples.csv)")