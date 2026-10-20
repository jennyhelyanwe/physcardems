"""plot_sweep.py: overview of every sample in a sweep, as far as each has got.

Rows: LV and RV (pressure-volume loop, pressure and volume against time), then the
prescribed Ta profile of each sample for endo, epi and mid, computed from its config
(time since local activation, one cycle). Every curve is coloured by that sample's LVEF
once the LV has entered IVR, and drawn grey before then. Samples with LVEF inside the
target range are drawn on top, with thicker lines.

    python3 scripts/plot_sweep.py runs/ta_lhs [--lvef 51 70]

Writes runs/<study>/sweep_overview.png. Safe to run while the sweep is going.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physcardems.plotting import ejection_metrics
from physcardems.prescribed_ta import CELLTYPES, celltype_profiles, diastolic_ta, ta_shape

ap = argparse.ArgumentParser()
ap.add_argument("study_dir", help="e.g. runs/ta_lhs")
ap.add_argument("--lvef", nargs=2, type=float, default=[51.0, 70.0], help="target LVEF range (%%)")
ap.add_argument("--out", default="sweep_overview.png")
ap.add_argument("--lat0", type=float, default=130.0,
                help="activation time (ms) at which the Ta panels start each profile, so they line up with "
                     "the pressure and volume traces; 130 ms is the earliest activation in rodero_05")
args = ap.parse_args()
study = Path(args.study_dir)
lo, hi = args.lvef

samples = []
for cfg_path in sorted(study.glob("sample_*/config.json")):
    sdir = cfg_path.parent
    cfg = json.loads(cfg_path.read_text())
    s = {"id": int(sdir.name.split("_")[1]), "cfg": cfg, "log": None, "lvef": np.nan}
    log = sdir / "log.csv"
    if log.exists():
        a = np.loadtxt(log, delimiter=",", skiprows=1, ndmin=2)
        if len(a) > 1:
            s["log"] = a
            edv, esv, ef, done = ejection_metrics(a[:, 5], a[:, 4])
            if done:
                s["lvef"] = ef
    samples.append(s)

n_started = sum(s["log"] is not None for s in samples)
efs = np.array([s["lvef"] for s in samples])
n_done = int(np.isfinite(efs).sum())
n_in = int(np.sum((efs >= lo) & (efs <= hi)))
finite = efs[np.isfinite(efs)]
norm = matplotlib.colors.Normalize(vmin=min(20.0, finite.min()) if finite.size else 20.0,
                                   vmax=max(80.0, finite.max()) if finite.size else 80.0)
cmap = plt.get_cmap("viridis")


def style(s):
    ef = s["lvef"]
    if not np.isfinite(ef):
        return dict(color="0.75", lw=0.6, alpha=0.6, zorder=1)
    inside = lo <= ef <= hi
    return dict(color=cmap(norm(ef)), lw=1.6 if inside else 0.7, alpha=1.0 if inside else 0.55,
                zorder=3 if inside else 2)


fig, ax = plt.subplots(3, 3, figsize=(18, 14), gridspec_kw={"hspace": 0.4, "wspace": 0.25})
order = sorted(samples, key=lambda s: (np.isfinite(s["lvef"]), lo <= s["lvef"] <= hi))  # good ones last
for s in order:
    a = s["log"]
    if a is None:
        continue
    st = style(s)
    t = a[:, 0]
    for r, (iv, ip) in enumerate(((5, 6), (9, 10))):
        ax[r, 0].plot(a[:, iv], a[:, ip], **st)
        ax[r, 1].plot(t, a[:, ip], **st)
        ax[r, 2].plot(t, a[:, iv], **st)

# prescribed Ta from each sample's config (no simulation output needed)
for s in order:
    cfg = s["cfg"]
    pcl = float(cfg["time"]["pcl_ms"])
    prof = celltype_profiles(cfg["active"], pcl)
    ta_d = diastolic_ta(cfg["active"], prof)
    # tau = np.linspace(0.0, pcl, 400)
    t_sim = np.linspace(0.0, pcl, 801)
    tau = t_sim - args.lat0 - float(cfg.get("activation", {}).get("lat_shift_ms", 0.0))
    st = style(s)
    for c, name in enumerate(CELLTYPES.values()):
        p = prof[name]
        ta = ta_d[name] + (p["Ta_peak_kpa"] - ta_d[name]) * ta_shape(
            tau, p["t_peak_ms"], p["a_rise"], p["sigma_decay_ms"], pcl)
        shape = np.where(tau >= 0.0, ta_shape(np.clip(tau, 0.0, None), p["t_peak_ms"], p["a_rise"],
                                              p["sigma_decay_ms"], pcl), 0.0)
        ax[2, c].plot(t_sim, ta_d[name] + (p["Ta_peak_kpa"] - ta_d[name]) * shape, **st)
        # ax[2, c].plot(tau, ta, **st)

for r, side in enumerate(("LV", "RV")):
    ax[r, 0].set(title=f"{side} pressure-volume loop", xlabel="volume (mL)", ylabel="pressure (kPa)")
    ax[r, 1].set(title=f"{side} pressure", xlabel="t (ms)", ylabel="pressure (kPa)")
    ax[r, 2].set(title=f"{side} volume", xlabel="t (ms)", ylabel="volume (mL)")
for c, name in enumerate(CELLTYPES.values()):
    # ax[2, c].set(title=f"Prescribed Ta, {name}", xlabel="time since activation (ms)", ylabel="Ta (kPa)")
    ax[2, c].set(title=f"Prescribed Ta, {name} (activated at {args.lat0:g} ms)", xlabel="t (ms)",
                 ylabel="Ta (kPa)")
# for a_ in ax.flat:
#     a_.grid(True, alpha=0.3)

t_max = max([s["log"][-1, 0] for s in samples if s["log"] is not None] + [args.lat0 + 400.0])
for a_ in (ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2], *ax[2, :]):
    a_.set_xlim(0.0, t_max)
for a_ in ax.flat:
    a_.grid(True, alpha=0.3)

sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cb.set_label("LVEF (%)")
cb.ax.axhspan(lo, hi, color="w", alpha=0.35)
fig.suptitle(f"{study}: {len(samples)} samples, {n_started} started, {n_done} past ejection, "
             f"{n_in} with LVEF {lo:g} to {hi:g}% (thick); grey = not past ejection yet", fontsize=13)
fig.savefig(study / args.out, dpi=110, bbox_inches="tight")
print(f"wrote {study / args.out}: {n_started} started, {n_done} past ejection, {n_in} in LVEF range")