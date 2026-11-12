"""Compare the same run in two codes: a physcardems log.csv and a simcardemsx rodero_05 log.csv.

    pcepy scripts/compare_simcardemsx.py runs/rodero_05/em_tref7 runs/rodero_05/simcardemsx_em_tref7
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASES = ["PRELOAD", "IVC", "EJECTION", "IVR", "FILLING"]
CAV = ("lv", "rv")


def load(run_dir, code):
    d = pd.read_csv(Path(run_dir) / "log.csv")
    d.columns = [c.strip().lstrip("#").strip() for c in d.columns]
    # Henrik's example logs the phase after the step's switch as next_phase_*; physcardems'
    # own log (old and new back end) already calls that phase_*.
    for c in CAV:
        if f"next_phase_{c}" in d.columns:
            d[f"phase_{c}"] = d[f"next_phase_{c}"]
    return d

def entries(ph, p):
    """Row indices where the phase becomes p."""
    prev = np.r_[-1, ph[:-1]]
    return np.flatnonzero((ph == p) & (prev != p))


def metrics(d, c):
    t = d["t_ms"].to_numpy()
    ph = d[f"phase_{c}"].to_numpy().astype(int)
    V, P = d[f"V_{c}_mL"].to_numpy(), d[f"P_{c}_kPa"].to_numpy()
    out = {"Pmax_kPa": P.max()}
    ivc = entries(ph, 1)
    if len(ivc):
        i = ivc[-1]
        edv, esv = V[i], V[i:].min()
        out.update(EDV_mL=edv, ESV_mL=esv, SV_mL=edv - esv, EF_pct=100 * (edv - esv) / edv)
    for p in range(1, 5):
        e = entries(ph, p)
        out[f"t_{PHASES[p]}_ms"] = t[e[0]] if len(e) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("physcardems_dir", help="physcardems run folder with log.csv")
    ap.add_argument("simcardemsx_dir", help="simcardemsx output folder with log.csv")
    ap.add_argument("--out", default="runs/rodero_05/compare_simcardemsx_em_tref7")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {"physcardems": load(a.physcardems_dir, "physcardems"),
            "simcardemsx": load(a.simcardemsx_dir, "simcardemsx")}

    rows = []
    for c in CAV:
        m = {k: metrics(d, c) for k, d in runs.items()}
        for key in m["physcardems"]:
            x, y = m["physcardems"][key], m["simcardemsx"].get(key, np.nan)
            rows.append({"cavity": c.upper(), "metric": key,
                         "physcardems": x, "simcardemsx": y, "diff": y - x})
    tab = pd.DataFrame(rows)

    # largest pointwise differences over the time both runs cover (both use 2 ms steps)
    a_, b_ = runs["physcardems"], runs["simcardemsx"]
    tmax = min(a_.t_ms.max(), b_.t_ms.max())
    tt = a_.t_ms[a_.t_ms <= tmax].to_numpy()
    diffs = {}
    for col in [f"V_{c}_mL" for c in CAV] + [f"P_{c}_kPa" for c in CAV] + ["Ta_max_kPa"]:
        diffs[f"max_abs_diff_{col}"] = float(np.max(np.abs(
            np.interp(tt, b_.t_ms, b_[col]) - np.interp(tt, a_.t_ms, a_[col]))))
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\ncompared up to t = {tmax:.0f} ms")
    print(json.dumps(diffs, indent=1))
    tab.to_csv(out / "metrics.csv", index=False)
    (out / "pointwise.json").write_text(json.dumps(diffs, indent=1))

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    style = {"physcardems": dict(c="k", lw=2), "simcardemsx": dict(c="C1", lw=1.5, ls="--")}
    for k, d in runs.items():
        for j, c in enumerate(CAV):
            ax[j, 0].plot(d[f"V_{c}_mL"], d[f"P_{c}_kPa"], label=k, **style[k])
            ax[j, 1].plot(d.t_ms, d[f"P_{c}_kPa"], **style[k])
            ax[j, 2].plot(d.t_ms, d[f"V_{c}_mL"], **style[k])
    for j, c in enumerate(CAV):
        ax[j, 0].set(title=f"{c.upper()} PV loop", xlabel="V (mL)", ylabel="P (kPa)")
        ax[j, 1].set(title=f"{c.upper()} pressure", xlabel="t (ms)", ylabel="P (kPa)")
        ax[j, 2].set(title=f"{c.upper()} volume", xlabel="t (ms)", ylabel="V (mL)")
    ax[0, 0].legend()
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "compare_pv.png", dpi=130)

    fig, x = plt.subplots(figsize=(6, 3.5))
    for k, d in runs.items():
        x.plot(d.t_ms, d["Ta_max_kPa"], label=k, **style[k])
    x.set(xlabel="t (ms)", ylabel="max Ta (kPa)", title="Peak active tension")
    x.legend()
    x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "compare_ta.png", dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()