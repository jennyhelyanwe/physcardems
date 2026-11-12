"""wearables_demo.py: simulated wearable signals for one beat of a physcardems run.

Usage (from the repository root, inside the container):
    python3 scripts/wearables_demo.py --synthetic
    python3 scripts/wearables_demo.py runs/<case>/<run> --lat0 <first activation, ms>
Works with LV ellipsoid and BiV logs (systemic side only for the BiV). If the run folder holds
a pseudo-ECG csv (a file name containing "ecg"), lead II is shown on top.
Writes wearables.png and wearables.json to the run folder (the current folder for --synthetic),
plus traces.js for demo.html when a run folder is given.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physcardems.wearables import MMHG, beat_from_log, load_run, synthetic_beat, wearable_signals


def load_ecg(run_dir):
    """First csv in the run folder whose name contains 'ecg': (file name, t_ms, lead values, lead)."""
    files = sorted(p for p in Path(run_dir).glob("*.csv") if "ecg" in p.name.lower())
    if not files:
        return None
    e = np.genfromtxt(files[0], delimiter=",", names=True)
    names = e.dtype.names
    lead = "II" if "II" in names else names[1]
    return files[0].name, np.asarray(e[names[0]], float), np.asarray(e[lead], float), lead


PHASES = ["diastolic filling", "isovolumic contraction", "ejection", "isovolumic relaxation", "filling"]


def export_traces(run_dir, beat, sig, period_ms, comp, dt_out=2.0):
    """Write traces.js (window.TRACES = {...}) for demo.html.

    Everything sits on one grid of absolute simulation time, 0..period_ms, the same clock as
    log.csv, pseudo_ecg.csv and the movie, so the page maps video time straight onto it.
    """
    run_dir = Path(run_dir)
    log = np.genfromtxt(run_dir / "log.csv", delimiter=",", names=True)
    t = np.arange(0.0, period_ms + 0.5 * dt_out, dt_out)
    kpa = 7.50062  # kPa -> mmHg
    lin = lambda col: np.interp(t, log["t_ms"], log[col])
    out = dict(t_ms=t, period_ms=period_ms)
    out["V_lv"], out["P_lv"] = lin("V_lv_mL"), lin("P_lv_kPa") * kpa
    out["V_rv"], out["P_rv"] = lin("V_rv_mL"), lin("P_rv_kPa") * kpa
    out["phase"] = np.interp(t, log["t_ms"], log["phase_lv"]).round().astype(int)
    out["phase_names"] = PHASES

    # wearable signals live on beat["t_s"], measured from first activation: shift to absolute time
    t_abs = beat["t_s"] * 1e3 + beat["t_act_ms"]
    per = lambda y: np.interp(t, t_abs, np.asarray(y, float), period=period_ms)
    norm = lambda y: (y - y.min()) / (y.max() - y.min())
    out["ppg_finger"], out["ppg_wrist"] = norm(per(sig["finger"])), norm(per(sig["wrist"]))
    out["p_aortic"] = per(sig["p_root"]) / MMHG
    out["p_radial"] = per(sig["flows"]["r_radial"]["p_out"]) / MMHG
    out["icg"] = per(sig["mdzdt"])
    out["t_act_ms"] = beat["t_act_ms"]

    ecg_files = sorted(p for p in run_dir.glob("*.csv") if "ecg" in p.name.lower())
    if ecg_files:
        e = np.genfromtxt(ecg_files[0], delimiter=",", names=True)
        for lead in ("II", "V1", "V5"):
            if lead in e.dtype.names:
                out["ecg_" + lead] = np.interp(t, e[e.dtype.names[0]], e[lead])

    edv, esv = log["V_lv_mL"].max(), log["V_lv_mL"].min()
    wall = log["wall_s"][np.isfinite(log["wall_s"])]
    out["stats"] = dict(
        EF_lv=100 * (edv - esv) / edv, SV_lv=edv - esv, EDV_lv=edv,
        Pmax_lv=np.nanmax(log["P_lv_kPa"]) * kpa,
        dt_ms=float(np.median(np.diff(log["t_ms"]))), n_steps=len(log),
        newton_mean=float(np.nanmean(log["newton_its"])), wall_h=float(wall.sum() / 3600),
        # compliance=comp, **{k: float(v) for k, v in sig["features"].items()})
        compliance = comp,
        ** {k: float(v) for k, v in sig["features"].items() if np.ndim(v) == 0})
    def js(v):
        if isinstance(v, np.ndarray):
            return [round(float(x), 5) for x in v]
        if isinstance(v, dict):
            return {k: js(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    path = run_dir / "traces.js"
    path.write_text("window.TRACES = " + json.dumps(js(out)) + ";\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", help="run folder with log.csv and config_used.json")
    ap.add_argument("--synthetic", action="store_true", help="use a stand-in beat, no run needed")
    ap.add_argument("--lat0", type=float, default=0.0, help="time of first activation in beat 1 (ms)")
    ap.add_argument("--stiffness-scale", type=float, default=1.0, help="arterial wall stiffness multiplier")
    ap.add_argument("--compliance", type=float, default=None,
                    help="total arterial compliance (mL/mmHg); default from the config, else 1.5")
    args = ap.parse_args()

    ecg = None
    if args.synthetic:
        beat, out_dir, label, comp = synthetic_beat(), Path("."), "synthetic beat", args.compliance or 1.5
    else:
        log, cfg = load_run(args.run)
        pcl = cfg.get("time", {}).get("pcl_ms", 800.0)
        wk = cfg.get("circulation", {}).get("lv", {}).get("windkessel", {})
        comp = args.compliance or wk.get("compliance_ml_per_mmhg", 1.5)
        beat, out_dir, label = beat_from_log(log, pcl, args.lat0), Path(args.run), Path(args.run).name
        ecg = load_ecg(args.run)
    sig = wearable_signals(beat, comp, args.stiffness_scale)
    feat = sig["features"]

    # two identical cycles, so the waveforms read as a recording
    T = beat["period_s"] * 1e3
    tt = np.concatenate([beat["t_s"] * 1e3, beat["t_s"] * 1e3 + T])
    two = lambda y: np.concatenate([y, y])
    norm = lambda y: (y - y.min()) / (y.max() - y.min())

    nrow = 5 if ecg else 4
    fig, axs = plt.subplots(nrow, 1, figsize=(9, 2.6 * nrow + 0.6), sharex=True)
    for a in axs:
        for k in (0, 1):
            a.axvspan(beat["t_open_s"] * 1e3 + k * T, beat["t_close_s"] * 1e3 + k * T, color="0.92", zorder=0)
            a.axvline(k * T, color="0.5", lw=0.8, ls=":")
        a.grid(True, alpha=0.3)
    if ecg:
        name, t_e, v_e, lead = ecg
        rel = t_e - beat["t_act_ms"]
        keep = (rel >= 0) & (rel < 2 * T)
        axs[0].plot(rel[keep], v_e[keep], color="k", lw=1)
        axs[0].set(ylabel=f"pseudo-ECG\nlead {lead}", title=f"ECG from {name}")
        ax = axs[1:]
    else:
        ax = axs

    ax[0].plot(tt, two(sig["p_root"]) / MMHG, label="aortic root (1D network)")
    ax[0].plot(tt, two(sig["flows"]["r_radial"]["p_out"]) / MMHG, label="radial artery (1D network)")
    if np.isfinite(beat["p_lv"]).any():
        ax[0].plot(tt, two(beat["p_lv"]) / MMHG, color="k", lw=1, label="LV (heart model)")
    ax[0].set(ylabel="pressure (mmHg)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[1].plot(tt, two(beat["q"]) * 1e6, color="k")
    ax[1].set(ylabel="aortic flow (mL/s)")
    ax[2].plot(tt, two(sig["mdzdt"]), color="C2")
    for name_pt, tp in zip("BCX", sig["icg"]):
        y = np.interp(tp, beat["t_s"], sig["mdzdt"])
        ax[2].plot(tp * 1e3, y, "o", color="C2")
        ax[2].annotate(name_pt, (tp * 1e3, y), textcoords="offset points", xytext=(4, 6))
    ax[2].set(ylabel="impedance cardiogram\n-dZ/dt (ohm/s)")
    for y, f, lab, c in ((sig["finger"], sig["feet"][1], "finger", "C3"),
                         (sig["wrist"], sig["feet"][2], "wrist", "C1")):
        ax[3].plot(tt, two(norm(y)), color=c, label=f"PPG {lab}")
        ax[3].axvline(f * 1e3, color=c, lw=0.8, ls="--")
    ax[3].set(ylabel="PPG (normalised)", xlabel="time since first activation (ms)")
    ax[3].legend(fontsize=8, loc="upper right")

    summary = (f"PAT finger {feat['PAT_finger_ms']:.0f} ms, PTT finger {feat['PTT_finger_ms']:.0f} ms   |   "
               f"PEP icg {feat['PEP_icg_ms']:.0f} (model {feat['PEP_model_ms']:.0f}) ms, "
               f"LVET icg {feat['LVET_icg_ms']:.0f} (model {feat['LVET_model_ms']:.0f}) ms")
    fig.suptitle(f"Simulated wearable signals: {label}\nGrey: ejection (heart model). "
                 f"Dotted: first activation (Q onset)\n{summary}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "wearables.png", dpi=120)

    record = dict(features=feat, notes=sig["notes"], lat0_ms=args.lat0,
                  stiffness_scale=args.stiffness_scale, compliance_ml_per_mmhg=comp)
    (out_dir / "wearables.json").write_text(json.dumps(record, indent=2, default=float))
    if not args.synthetic:
        export_traces(args.run, beat, sig, pcl, comp)
    for k, v in feat.items():
        print(f"{k:24s} {np.round(v, 1)}")
    for note in sig["notes"]:
        print("note:", note)
    if not args.synthetic and ecg is None:
        print("no ECG csv found in the run folder; ECG panel skipped")
    print(f"wrote {out_dir / 'wearables.png'} and wearables.json")


if __name__ == "__main__":
    main()