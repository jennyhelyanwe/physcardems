"""pacing.py: steady-state initial conditions for the 3D run.

Paces the full (unsplit) ToR-ORd-Land single cell, isometric (lambda = 1), for endo, epi
and mid cells until the beat-to-beat change in peak calcium and the slow ion concentrations
falls below a tolerance. The final states become the initial conditions of every node of
that cell type in the 3D simulation.

The states depend on the cell-model parameters (ionic conductances, Land calcium
sensitivity, ...), so they are stored in a folder named after a hash of those parameters.
Mechanics-only Land parameters (Tref, Beta0, Tot_A) do not change the states and are left
out of the hash. lv_simulation.py calls ensure_steady_state() at start-up: it reuses a
matching folder if one exists, and paces a new one otherwise (a few minutes).
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from physcardems import parameters as mp

CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}  # ToR-ORd convention


def _load_full_model(module_file: Path, odefile: Path):
    import gotranx
    if not module_file.exists():
        ode = gotranx.load_ode(odefile)
        code = gotranx.cli.gotran2py.get_code(ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen])
        module_file.write_text(code)
    spec = importlib.util.spec_from_file_location("torord_land_full", module_file)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def _apd90(t, v):
    vmax, vmin = v.max(), v.min()
    v90 = vmax - 0.9 * (vmax - vmin)
    dep = int(np.argmax(v > v90))
    peak = int(np.argmax(v))
    rep = peak + int(np.argmax(v[peak:] < v90))
    return t[rep] - t[dep]


def pace_steady_state(outdir, land, ep_overrides, ep_scales, pcl_ms=800.0, dt_ms=0.05,
                      max_beats=300, min_beats=20, tol=1e-3, odefile=mp.ODEFILE, log=print):
    """Pace the three cell types to steady state and write celltype<c>_<name>.json (+ .npz, diagnostic.png)."""
    import numba
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    phash = mp.parameter_hash(land, ep_overrides, ep_scales)
    model = _load_full_model(outdir / "torord_land_full.py", Path(odefile))
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

    nsteps = int(round(pcl_ms / dt_ms))
    rec_every = max(1, int(round(1.0 / dt_ms)))
    results = {}
    for ct, name in CELLTYPES.items():
        y = model.init_state_values()
        p = model.init_parameter_values(celltype=float(ct), i_Stim_Period=pcl_ms, i_Stim_Start=0.0,
                                        lmbda=1.0, dLambda=0.0)
        mp.apply_to_ode_parameters(p, model, land, ep_overrides, ep_scales)
        t, history, converged, t0 = 0.0, [], False, time.time()
        for beat in range(1, max_beats + 1):
            y, t, rec = run_beat(y, t, nsteps, dt_ms, p, rec_every)
            history.append(np.array([rec[:, ica].max(), y[ina], y[ik], y[insr]]))
            if beat > min_beats and np.all(np.abs(history[-1] - history[-2]) / np.abs(history[-2]) < tol):
                converged = True
                break
        tr = np.arange(rec.shape[0]) * rec_every * dt_ms
        Ta = np.array([model.monitor_values(tt, s, p)[iTa] for tt, s in zip(tr, rec)])
        bio = dict(APD90_ms=float(_apd90(tr, rec[:, iv])), cai_peak_uM=float(rec[:, ica].max() * 1e3),
                   cai_rest_uM=float(rec[:, ica].min() * 1e3), Ta_peak_kPa=float(Ta.max()),
                   Ta_rest_kPa=float(Ta.min()))
        log(f"steady state {name}: {beat} beats, converged={converged}, {time.time() - t0:.0f} s, "
            f"APD90 {bio['APD90_ms']:.0f} ms, Cai {bio['cai_rest_uM']:.3f}-{bio['cai_peak_uM']:.3f} uM")
        (outdir / f"celltype{ct}_{name}.json").write_text(json.dumps({
            "celltype": ct, "name": name, "pcl_ms": pcl_ms, "dt_ms": dt_ms, "beats": beat,
            "converged": converged, "tol": tol, "odefile": str(odefile), "parameter_hash": phash,
            "land": land, "ep_overrides": ep_overrides, "ep_scales": ep_scales, "biomarkers": bio,
            "states": {s: float(y[model.state_index(s)]) for s in state_names},
        }, indent=2))
        np.savez(outdir / f"celltype{ct}_{name}_last_beat.npz", t=tr, v=rec[:, iv], cai=rec[:, ica], Ta=Ta,
                 history=np.array(history))
        results[name] = dict(t=tr, v=rec[:, iv], cai=rec[:, ica], Ta=Ta)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in results.items():
        ax[0].plot(r["t"], r["v"], label=name)
        ax[1].plot(r["t"], r["cai"] * 1e3, label=name)
        ax[2].plot(r["t"], r["Ta"], label=name)
    for a, title in zip(ax, ("V (mV)", "Cai (uM)", "Ta (kPa), isometric")):
        a.set(title=title, xlabel="time in beat (ms)")
        a.legend()
    fig.suptitle(f"steady state, parameter hash {phash}")
    fig.tight_layout()
    fig.savefig(outdir / "diagnostic.png", dpi=100)
    plt.close(fig)


def steady_state_ready(ssdir, phash) -> bool:
    for c, name in CELLTYPES.items():
        f = Path(ssdir) / f"celltype{c}_{name}.json"
        if not f.exists() or json.loads(f.read_text()).get("parameter_hash") != phash:
            return False
    return True


def ensure_steady_state(ssdir, land, ep_overrides, ep_scales, pcl_ms, log=print, wait_s=3600):
    """Reuse the steady state in ssdir if it matches the parameters, otherwise pace it.

    Safe when several runs with the same new parameters start together: the first one paces
    (holding a lock file), the others wait for it to finish.
    """
    ssdir = Path(ssdir)
    phash = mp.parameter_hash(land, ep_overrides, ep_scales)
    if steady_state_ready(ssdir, phash):
        return
    ssdir.mkdir(parents=True, exist_ok=True)
    lock = ssdir / ".pacing.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log(f"another run is pacing {ssdir}; waiting")
        t0 = time.time()
        while not steady_state_ready(ssdir, phash):
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"steady state in {ssdir} not ready after {wait_s} s (stale {lock}?)")
            time.sleep(10)
        return
    try:
        os.close(fd)
        log(f"no steady state for parameter hash {phash}: pacing single cells (a few minutes)")
        pace_steady_state(ssdir, land, ep_overrides, ep_scales, pcl_ms=pcl_ms, log=log)
    finally:
        lock.unlink(missing_ok=True)