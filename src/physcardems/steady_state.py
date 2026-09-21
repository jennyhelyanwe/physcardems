"""pace_steady_state.py

Paces the full (unsplit) ToR-ORd-Land model from the simcardemsx zeta-split .ode
to steady state for endo, epi and mid cells, isometric (lmbda = 1, dLambda = 0),
with the parameter set from model_parameters.py. Output goes to a folder named
after the parameter hash, which the 3D run checks.

Run with plain python3 (single process).
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import numba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gotranx

from physcardems import parameters as mp

CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}  # ToR-ORd convention

ap = argparse.ArgumentParser()
ap.add_argument("--odefile", default="/home/shared/simcardemsx/numerical_experiments/odefiles/ToRORd_dynCl_endo_zetasplit.ode")
ap.add_argument("--outdir", default=None)
ap.add_argument("--pcl", type=float, default=800.0)
ap.add_argument("--dt", type=float, default=0.05)
ap.add_argument("--max-beats", type=int, default=300)
ap.add_argument("--min-beats", type=int, default=20)
ap.add_argument("--tol", type=float, default=1e-3)
args = ap.parse_args()

land = mp.land_values()
phash = mp.parameter_hash(land)
outdir = Path(args.outdir) if args.outdir else Path(f"steady_state_pcl{int(args.pcl)}_{phash}")
outdir.mkdir(parents=True, exist_ok=True)
print(f"parameter hash {phash} -> {outdir}")

# ------------------------------------------------------------------ generate and load the full model
module_file = outdir / "torord_land_full.py"
if not module_file.exists():
    ode = gotranx.load_ode(args.odefile)
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen])
    module_file.write_text(code)
spec = importlib.util.spec_from_file_location("torord_land_full", module_file)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

state_names = sorted(model.state, key=model.state.get)
iv, ica, ina, ik, insr = (model.state_index(s) for s in ("v", "cai", "nai", "ki", "cansr"))
iTa = model.monitor_index("Ta")
grl = numba.njit(model.generalized_rush_larsen)


@numba.njit
def run_beat(y, t0, nsteps, dt, p, rec_every):
    rec = np.zeros((nsteps // rec_every, y.size))
    t = t0
    for i in range(nsteps):
        if i % rec_every == 0:
            rec[i // rec_every] = y
        y = grl(y, t, dt, p)
        t += dt
    return y, t, rec


def apd90(t, v):
    vmax, vmin = v.max(), v.min()
    v90 = vmax - 0.9 * (vmax - vmin)
    dep = int(np.argmax(v > v90))
    peak = int(np.argmax(v))
    rep = peak + int(np.argmax(v[peak:] < v90))
    return t[rep] - t[dep]


nsteps = int(round(args.pcl / args.dt))
rec_every = max(1, int(round(1.0 / args.dt)))  # record every 1 ms
results = {}

for ct, name in CELLTYPES.items():
    y = model.init_state_values()
    p = model.init_parameter_values(celltype=float(ct), i_Stim_Period=args.pcl, i_Stim_Start=0.0,
                                    lmbda=1.0, dLambda=0.0)
    applied = mp.apply_to_ode_parameters(p, model, land)
    if ct == 0:
        print("parameters applied:", applied)
    t = 0.0
    history = []
    converged = False
    t_start = time.time()
    for beat in range(1, args.max_beats + 1):
        y, t, rec = run_beat(y, t, nsteps, args.dt, p, rec_every)
        markers = np.array([rec[:, ica].max(), y[ina], y[ik], y[insr]])
        history.append(markers)
        if beat > args.min_beats:
            rel = np.abs(markers - history[-2]) / np.abs(history[-2])
            if np.all(rel < args.tol):
                converged = True
                break
    elapsed = time.time() - t_start

    tr = np.arange(rec.shape[0]) * rec_every * args.dt
    Ta = np.array([model.monitor_values(tt, s, p)[iTa] for tt, s in zip(tr, rec)])
    bio = dict(APD90_ms=float(apd90(tr, rec[:, iv])), Vmax_mV=float(rec[:, iv].max()),
               cai_peak_uM=float(rec[:, ica].max() * 1e3), Ta_peak_kPa=float(Ta.max()),
               Ta_time_to_peak_ms=float(tr[np.argmax(Ta)]))
    print(f"{name:4s}: {beat} beats, converged={converged}, {elapsed:.0f} s | "
          f"APD90 {bio['APD90_ms']:.0f} ms, Cai peak {bio['cai_peak_uM']:.3f} uM, "
          f"Ta peak {bio['Ta_peak_kPa']:.1f} kPa at {bio['Ta_time_to_peak_ms']:.0f} ms")

    (outdir / f"celltype{ct}_{name}.json").write_text(json.dumps({
        "celltype": ct, "name": name, "pcl_ms": args.pcl, "dt_ms": args.dt,
        "beats": beat, "converged": converged, "tol": args.tol,
        "odefile": args.odefile, "parameter_hash": phash, "land": land,
        "ep_overrides": mp.EP_OVERRIDES, "ep_scales": mp.EP_SCALES,
        "biomarkers": bio,
        "states": {s: float(y[model.state_index(s)]) for s in state_names},
    }, indent=2))
    np.savez(outdir / f"celltype{ct}_{name}_last_beat.npz",
             t=tr, v=rec[:, iv], cai=rec[:, ica], Ta=Ta, history=np.array(history))
    results[ct] = dict(t=tr, v=rec[:, iv], cai=rec[:, ica], Ta=Ta, history=np.array(history))

# ------------------------------------------------------------------ diagnostic figure
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for ct, r in results.items():
    lab = CELLTYPES[ct]
    ax[0, 0].plot(r["t"], r["v"], label=lab)
    ax[0, 1].plot(r["t"], r["cai"] * 1e3, label=lab)
    ax[1, 0].plot(r["t"], r["Ta"], label=lab)
    ax[1, 1].plot(np.arange(1, len(r["history"]) + 1), r["history"][:, 0] * 1e3, label=lab)
ax[0, 0].set_title("V (mV), last beat")
ax[0, 1].set_title("Cai (uM), last beat")
ax[1, 0].set_title("Ta (kPa), last beat, isometric")
ax[1, 1].set_title("Peak Cai (uM) per beat")
ax[1, 1].set_xlabel("beat")
for a in ax.flat:
    a.legend()
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.set_xlabel("time in beat (ms)")
fig.suptitle(f"parameter hash {phash}")
fig.tight_layout()
fig.savefig(outdir / "diagnostic.png")
print(f"saved to {outdir}")