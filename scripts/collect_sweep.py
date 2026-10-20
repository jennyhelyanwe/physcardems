"""collect_sweep.py: gather pressure-volume biomarkers from a sweep and rank the samples.

Reads each sample's log.csv (so unfinished or failed runs are reported too), computes
EDV, ESV, SV, EF, end-diastolic and peak pressure for both ventricles and the peak LV
ejection rate, joins them to the sampled parameters, and scores every sample against the
pressure-volume criteria in criteria.toml.

    python3 scripts/collect_sweep.py runs/ta_lhs_test

Writes runs/<study>/results.csv (one row per sample, best first) and prints the top rows.
Status: complete = both ventricles have entered IVR, so the first-beat biomarkers are final
(the run may still be going); running = not there yet; failed = the run ended before that.
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
ap.add_argument("--summary", action="store_true",
                help="also print a markdown summary of the completed samples, for configs/ta_sweep/NOTES.md")
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
    # bm["status"] = "complete" if (complete and finished) else ("failed" if finished else "running")
    # the first-beat biomarkers are final once both ventricles have entered IVR,
    # whether or not the run has reached t_end
    bm["status"] = "complete" if complete else ("failed" if finished else "running")
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

def md_table(df, index_name="id"):
    """Markdown table without extra dependencies; floats to 3 significant figures."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.3g}")
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join([index_name] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(i)] + [fmt(v) for v in row]) + " |"
              for i, row in zip(df.index, df.itertuples(index=False))]
    return "\n".join(lines)


counts = result["status"].value_counts().to_dict()
print(f"- {study}: {len(result)} samples, " + ", ".join(f"{v} {k}" for k, v in counts.items()))
print("- Targets: " + ", ".join(f"{c.biomarker} {c.lower:g}-{c.upper:g} {c.units}" for c in criteria))
print()
print(md_table(result[front[:-1]].head(args.top)))
print(f"\nwrote {study / 'results.csv'}")

if args.summary:
    comp = result[result["status"] == "complete"]
    print(f"\n- Collected: {len(comp)} of {len(result)} samples complete; "
          f"{int((comp['n_pass'] == comp['n_criteria']).sum())} pass all {len(criteria)} PV criteria.")
    print("\n| biomarker | target | min | median | max | n in range |")
    print("|---|---|---|---|---|---|")
    for c in criteria:
        x = comp[c.biomarker].astype(float)
        n_in = int(((x >= c.lower) & (x <= c.upper)).sum())
        print(f"| {c.biomarker} ({c.units}) | {c.lower:g}-{c.upper:g} | {x.min():.3g} | {x.median():.3g} | "
              f"{x.max():.3g} | {n_in}/{len(x)} |")
    best = comp.sort_values("rank_lv_first").head(3)
    print("\n- Best by lv_first: " + "; ".join(
        f"sample {i}: LVEF {r.LVEF:.1f}%, EDV {r.EDVL:.0f}, ESV {r.ESVL:.0f} mL, PmaxL {r.PmaxL:.1f} kPa, "
        f"RVEF {r.RVEF:.1f}%" for i, r in best.iterrows()))