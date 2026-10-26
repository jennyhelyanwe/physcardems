"""Compare LV runs (different meshes, parameters or settings), mid-run or finished.

    python3 scripts/compare_meshes_lv.py runs/lv_ellipsoid/baseline runs/lv_ellipsoid/baseline_tref9 \
        --out runs/lv_ellipsoid/tref_pair.png

Prints, per run: progress and time per step; V, P and max Ta at fixed times (end of preload and
the latest time every run has reached); and the biomarkers of each completed beat. The last run
given is the reference for the percentage differences. Runs are labelled by their last two path parts.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physcardems.plotting_lv import load_log, beat_biomarkers

ap = argparse.ArgumentParser()
ap.add_argument("rundirs", nargs="+")
ap.add_argument("--times", nargs="*", type=float, default=[120.0], help="extra comparison times (ms)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

runs = {}
for d in args.rundirs:
    p = Path(d) / "log.csv"
    if p.exists():
        runs["/".join(Path(d).parts[-2:])] = load_log(p)
    else:
        print(f"{d}: no log.csv yet")
names = list(runs)
ref = names[-1]

print(f"{'run':32s} {'t reached':>10s} {'s/step':>8s}")
for n, log in runs.items():
    print(f"{n:32s} {log['t_ms'][-1]:10.1f} {np.median(log['wall_s'][-20:]):8.1f}")

t_common = min(log["t_ms"][-1] for log in runs.values())
for t in sorted(set(args.times + [t_common])):
    print(f"\nat t = {t:.1f} ms" + ("  (latest common time)" if t == t_common else ""))
    vals = {}
    for n, log in runs.items():
        i = int(np.argmin(np.abs(log["t_ms"] - t)))
        if abs(log["t_ms"][i] - t) > 1e-6:
            continue
        vals[n] = (log["V_mL"][i], log["P_kPa"][i], log["Ta_max_kPa"][i], log["phase"][i])
    for n, (V, P, Ta, ph) in vals.items():
        d = f"  dV vs ref: {100 * (V - vals[ref][0]) / vals[ref][0]:+.1f}%" if ref in vals and n != ref else ""
        print(f"  {n:32s} V {V:7.2f} mL  P {P:6.2f} kPa  Ta max {Ta:7.1f} kPa  phase {int(ph)}{d}")

print("\ncompleted beats")
keys = ("EDV_mL", "ESV_mL", "LVEF_pct", "EDP_kPa", "Pmax_kPa", "dvdt_ejection_mL_s", "dpdt_max_kPa_s")
print(f"  {'run':32s} {'beat':>4s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
for n, log in runs.items():
    for b in beat_biomarkers(log):
        if b["complete"]:
            print(f"  {n:32s} {b['beat']:4d} " + " ".join(f"{b[k]:12.2f}" for k in keys))

if args.out:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for n, log in runs.items():
        ax[0].plot(log["t_ms"], log["V_mL"], label=n)
        ax[1].plot(log["t_ms"], log["P_kPa"], label=n)
        ax[2].plot(log["V_mL"], log["P_kPa"], label=n)
    ax[0].set(title="LV volume (mL)", xlabel="t (ms)")
    ax[1].set(title="LV pressure (kPa)", xlabel="t (ms)")
    ax[2].set(title="PV loop", xlabel="volume (mL)", ylabel="pressure (kPa)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")

    print('Blah')