"""collect_sweep.py: gather pressure-volume biomarkers from a sweep and rank the samples.

Reads each sample's log.csv (so unfinished or failed runs are reported too), computes
EDV, ESV, SV, EF, end-diastolic and peak pressure for both ventricles and the peak LV
ejection rate, joins them to the sampled parameters, and scores every sample against the
pressure-volume criteria in criteria.toml.

    python3 scripts/collect_sweep.py runs/ta_lhs_test

Writes runs/<study>/results.csv (one row per sample, best first) and prints the top rows.
Status: complete = first beat reached IVR in both ventricles and the run ended normally;
running = no metrics.json yet (still running, or stopped by the time limit);
failed = the run ended (metrics.json written) before completing ejection, e.g. a solver failure.
"""
import argparse
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from physcardems.criteria import load_criteria, evaluate
from physcardems.plotting import ejection_metrics

# criteria.toml biomarker -> how we compute it. dvdt_filling is left out: filling is
# volume-controlled at a prescribed rate, so it reflects the config, not the model.
USED = ["edvl", "esvl", "svl", "lvef", "edvr", "esvr", "svr", "rvef", "edpl", "pmaxl", "pmaxr",
        "dvdt_ejection"]
PROFILES = {
    "equal": {},
    "lv_first": {"tiers": [["lvef", "edvl", "esvl"], ["pmaxl", "edpl", "svl", "dvdt_ejection"],
                           ["rvef", "edvr", "esvr", "svr", "pmaxr"]]},
}


def biomarkers_from_log(log_path):
    a = np.loadtxt(log_path, delimiter=",", skiprows=1, ndmin=2)
    t = a[:, 0]
    out = {"t_reached_ms": float(t[-1]) if len(t) else np.nan}
    complete = True
    for side, iph, iv, ip in (("L", 4, 5, 6), ("R", 8, 9, 10)):
        V, P, ph = a[:, iv], a[:, ip], a[:, iph]
        edv, esv, ef, done = ejection_metrics(V, ph)
        complete &= done
        ivc = np.flatnonzero(ph == 1)
        out.update({f"EDV{side}": edv, f"ESV{side}": esv, f"SV{side}": edv - esv,
                    f"{side}VEF": ef, f"Pmax{side}": float(P.max()) if len(P) else np.nan,
                    f"EDP{side}": float(P[ivc[0]]) if ivc.size else np.nan})
        if side == "L":
            ej = ph == 2
            dvdt = -np.gradient(V, t) * 1e3                      # mL/s, positive when ejecting
            out["dvdt_ejection"] = float(dvdt[ej].max()) if ej.any() else np.nan
    return out, complete


ap = argparse.ArgumentParser()
ap.add_argument("study_dir", help="e.g. runs/ta_lhs_test")
ap.add_argument("--top", type=int, default=10)
args = ap.parse_args()
study = Path(args.study_dir)

params = pd.read_csv(study / "samples.csv").set_index("id")
rows = {}
for i in params.index:
    sdir = study / f"sample_{i:03d}"
    log = sdir / "log.csv"
    if not log.exists():
        rows[i] = {"status": "not started"}
        continue
    bm, complete = biomarkers_from_log(log)
    finished = (sdir / "metrics.json").exists()
    bm["status"] = "complete" if (complete and finished) else ("failed" if finished else "running")
    rows[i] = bm
bm = pd.DataFrame.from_dict(rows, orient="index")
bm.index.name = "id"

criteria, _ = load_criteria(files("physcardems") / "data" / "criteria.toml")
criteria = [c for c in criteria if c.name in USED]
for c in criteria:
    if c.biomarker not in bm.columns:
        bm[c.biomarker] = np.nan

# rank completed beats only; unfinished runs are listed after them with their values so far
done = bm.index[bm["status"] == "complete"]
cols = [c.biomarker for c in criteria]
if len(done):
    ranked, V = evaluate(bm.loc[done, cols], criteria, PROFILES, params=params.loc[done])
else:
    ranked = params.loc[[]]
rest = params.loc[params.index.difference(done)]
result = pd.concat([ranked, rest]).join(bm, rsuffix="_bm")
front = ["status", "LVEF", "EDVL", "ESVL", "SVL", "PmaxL", "EDPL", "dvdt_ejection", "RVEF", "PmaxR",
         "n_pass", "n_criteria", "score_equal", "rank_equal", "rank_lv_first", "t_reached_ms"]
result = result.reindex(columns=front + [c for c in result.columns if c not in front])
result.to_csv(study / "results.csv")

counts = result["status"].value_counts().to_dict()
print(f"{study}: {len(result)} samples, " + ", ".join(f"{v} {k}" for k, v in counts.items()))
print("targets: " + ", ".join(f"{c.biomarker} {c.lower:g}-{c.upper:g} {c.units}" for c in criteria))
with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
    print(result[front[:-1]].head(args.top))
print(f"wrote {study / 'results.csv'}")