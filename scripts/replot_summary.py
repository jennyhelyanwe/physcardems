"""Redraw summary.png from a finished (or running) run folder, without re-simulating.

Usage:
    python3 scripts/replot_summary.py runs/rodero_05/em_tref7 [--out summary_ef.png]
"""
import argparse
from pathlib import Path

import numpy as np

from physcardems.plotting import plot_summary, load_ecg_csv

ap = argparse.ArgumentParser()
ap.add_argument("rundir", help="run output folder containing log.csv")
ap.add_argument("--out", default="summary.png", help="file name inside rundir")
args = ap.parse_args()
rundir = Path(args.rundir)

arr = np.loadtxt(rundir / "log.csv", delimiter=",", skiprows=1, ndmin=2)
te, ve, leads = load_ecg_csv(rundir / "pseudo_ecg.csv")
plot_summary(arr, rundir / args.out, title=str(rundir), ecg_t=te, ecg_v=ve, ecg_leads=leads)
print(f"wrote {rundir / args.out}")