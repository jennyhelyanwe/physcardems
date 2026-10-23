"""Redraw summary.png and print beat-by-beat biomarkers for an LV-only run folder,
without re-simulating. Works on finished or still-running runs.

Usage:
    python3 scripts/replot_summary_lv.py runs/lv_ellipsoid_h4/baseline_h4 [--out summary.png]
"""
import argparse
import json
from pathlib import Path

from physcardems.plotting_lv import load_log, plot_summary, load_ecg_csv, beat_biomarkers

ap = argparse.ArgumentParser()
ap.add_argument("rundir", help="run output folder containing log.csv")
ap.add_argument("--out", default="summary.png", help="file name inside rundir")
args = ap.parse_args()
rundir = Path(args.rundir)

log = load_log(rundir / "log.csv")
te, ve, leads = load_ecg_csv(rundir / "pseudo_ecg.csv")
plot_summary(log, rundir / args.out, title=str(rundir), ecg_t=te, ecg_v=ve, ecg_leads=leads)
print(json.dumps(beat_biomarkers(log), indent=2))
print(f"wrote {rundir / args.out}")