"""Table of LV biomarkers for every run in a study folder (finished or still running),
using the last complete beat of each, with a pass count against the LV criteria and
Spearman correlations between the sampled parameters and the biomarkers.

    python3 scripts/collect_lv.py runs/lv_lhs1

Writes runs/<study>/results.csv, sorted by pass count, then by |LVEF - 60|.
"""
import argparse
import csv
import tomllib
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from physcardems.plotting_lv import load_log, beat_biomarkers

TARGETS = {"EDV_mL": "edvl", "ESV_mL": "esvl", "SV_mL": "svl", "LVEF_pct": "lvef", "EDP_kPa": "edpl",
           "Pmax_kPa": "pmaxl", "dvdt_ejection_mL_s": "dvdt_ejection", "dpdt_max_kPa_s": "dpdt_max"}

ap = argparse.ArgumentParser()
ap.add_argument("study")
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()
study = Path(args.study)
with open(files("physcardems") / "data" / "criteria.toml", "rb") as f:
    crit = tomllib.load(f)["criteria"]
ranges = {k: (crit[c]["lower"], crit[c]["upper"]) for k, c in TARGETS.items()}

samples, factors = {}, []
for fname in ("samples.csv", "grid.csv"):
    if (study / fname).exists():
        with open(study / fname) as f:
            reader = csv.DictReader(f)
            factors = [c for c in reader.fieldnames if c != "name"]
            samples = {r["name"]: {k: float(v) for k, v in r.items() if k != "name"} for r in reader}
        break

rows = []
for name in sorted(samples) or sorted(p.parent.name for p in study.glob("*/log.csv")):
    log_path = study / name / "log.csv"
    row = {"name": name, **samples.get(name, {})}
    if not log_path.exists():
        row["status"] = "not started"
        rows.append(row)
        continue
    log = load_log(log_path)
    done = [b for b in beat_biomarkers(log) if b["complete"]]
    row.update({"status": f"{len(done)} beat(s)" if done else "running/failed",
                "t_reached_ms": float(log["t_ms"][-1])})
    if done:
        b = done[-1]
        row.update({k: b[k] for k in TARGETS})
        row["n_pass"] = sum(lo <= b[k] <= hi for k, (lo, hi) in ranges.items())
    rows.append(row)

rows.sort(key=lambda r: (-r.get("n_pass", -1), abs(r.get("LVEF_pct", 1e9) - 60)))
cols = list(dict.fromkeys(k for r in rows for k in r))
with open(study / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

counts = {}
for r in rows:
    counts[r["status"]] = counts.get(r["status"], 0) + 1
print(f"{study}: {len(rows)} runs, " + ", ".join(f"{v} {k}" for k, v in counts.items()))
print("targets: " + ", ".join(f"{k} {lo:g}-{hi:g}" for k, (lo, hi) in ranges.items()))
show = ["name", "n_pass", *TARGETS]
print("\n" + " | ".join(show))
for r in rows[:args.top]:
    print(" | ".join(f"{r[c]:.3g}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in show))

done = [r for r in rows if "n_pass" in r]
if len(done) >= 8 and factors:
    print(f"\nSpearman correlation, {len(done)} complete runs (rows: factors, columns: biomarkers)")
    short = {k: k.split(".")[-1][:14] for k in factors}
    print(f"{'':16s}" + "".join(f"{b[:10]:>11s}" for b in TARGETS))
    for k in factors:
        x = [r[k] for r in done]
        rhos = [spearmanr(x, [r[b] for r in done])[0] for b in TARGETS]
        print(f"{short[k]:16s}" + "".join(f"{rho:11.2f}" for rho in rhos))
print(f"\nwrote {study / 'results.csv'}")