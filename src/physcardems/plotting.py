"""plotting.py: the run summary figure, shared by the run scripts and replot_summary.py.

Columns of log.csv (see LOG_HEADER):
  0 t_ms, 1 Ta_max_kPa, 2 lmbda_min, 3 lmbda_max, 4 phase_lv, 5 V_lv_mL, 6 P_lv_kPa, 7 Pc_lv_kPa,
  8 phase_rv, 9 V_rv_mL, 10 P_rv_kPa, 11 Pc_rv_kPa, 12 newton_its, 13 J_min, 14 J_max, 15 n_inverted,
  16 v_min, 17 v_max, 18 wall_ep_s, 19 wall_s, 20 Q_lv_mL_s, 21 Q_rv_mL_s
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# same as physcardems.cycle.PHASE_NAMES; copied so this module imports without dolfinx
PHASE_NAMES = {0: "PRELOAD", 1: "IVC", 2: "EJECTION", 3: "IVR", 4: "FILLING"}

LOG_HEADER = ("t_ms,Ta_max_kPa,lmbda_min,lmbda_max,phase_lv,V_lv_mL,P_lv_kPa,Pc_lv_kPa,"
              "phase_rv,V_rv_mL,P_rv_kPa,Pc_rv_kPa,newton_its,J_min,J_max,n_inverted,v_min,v_max,"
              "wall_ep_s,wall_s,Q_lv_mL_s,Q_rv_mL_s")
LEAD_ROWS = [["I", "II", "III", "aVR", "aVL", "aVF"], ["V1", "V2", "V3", "V4", "V5", "V6"]]


def ejection_metrics(V, phase):
    """EDV at entry to IVC (phase 1), ESV at entry to IVR (phase 3) of the first beat.

    Returns (edv, esv, ef_percent, complete). Mid-ejection, ESV is the minimum so far and
    complete is False. Returns NaNs before the first IVC.
    """
    ivc = np.flatnonzero(phase == 1)
    if ivc.size == 0:
        return np.nan, np.nan, np.nan, False
    edv = V[ivc[0]]
    ivr = np.flatnonzero((phase == 3) & (np.arange(len(V)) > ivc[0]))
    complete = bool(ivr.size)
    esv = V[ivr[0]] if complete else V[ivc[0]:].min()
    return edv, esv, 100.0 * (edv - esv) / edv, complete


def ef_text(V, phase):
    edv, esv, ef, complete = ejection_metrics(V, phase)
    if np.isnan(edv):
        return "EF n/a (no IVC yet)"
    tag = "" if complete else " (ejecting)"
    return f"EF {ef:.1f}%{tag}  (EDV {edv:.1f}, ESV {esv:.1f}, SV {edv - esv:.1f} mL)"


def plot_summary(arr, path, title="", ecg_t=None, ecg_v=None, ecg_leads=None):
    arr = np.atleast_2d(arr)
    t = arr[:, 0]
    fig = plt.figure(figsize=(20, 14))
    has_ecg = ecg_t is not None and len(ecg_t) > 1
    gs = fig.add_gridspec(4, 12, height_ratios=[1.0, 1.0, 0.55, 0.55], hspace=0.55, wspace=0.9)
    ax = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            ax[r, c] = fig.add_subplot(gs[r, 4 * c:4 * c + 4])

    ax[0, 0].plot(t, arr[:, 6], label="LV")
    ax[0, 0].plot(t, arr[:, 10], label="RV")
    ax[0, 0].plot(t, arr[:, 7], "--", label="P_c (sys)")
    ax[0, 0].plot(t, arr[:, 11], "--", label="P_c (pul)")
    ax[0, 0].set_title("Pressure (kPa)")
    ax[0, 1].plot(t, arr[:, 5], label="LV volume")
    ax[0, 1].plot(t, arr[:, 9], label="RV volume")
    axq = ax[0, 1].twinx()
    axq.plot(t, arr[:, 20], "C0:", label="LV outflow")
    axq.plot(t, arr[:, 21], "C1:", label="RV outflow")
    axq.set_ylabel("outflow (mL/s)")
    axq.legend(loc="lower right")
    ax[0, 1].set_title("Volume (mL) and outflow (mL/s)")
    ax[0, 2].plot(arr[:, 5], arr[:, 6], label="LV")
    ax[0, 2].plot(arr[:, 9], arr[:, 10], label="RV")
    ax[0, 2].set_title(f"PV loop (kPa vs mL)\nLV {ef_text(arr[:, 5], arr[:, 4])}\n"
                       f"RV {ef_text(arr[:, 9], arr[:, 8])}", fontsize=10)
    ax[1, 0].plot(t, arr[:, 1], "k", label="max Ta, myocardium (kPa)")
    axp = ax[1, 0].twinx()
    axp.step(t, arr[:, 4], where="post", label="LV phase")
    axp.step(t, arr[:, 8], where="post", label="RV phase")
    axp.set_yticks(list(PHASE_NAMES))
    axp.set_yticklabels(list(PHASE_NAMES.values()))
    ax[1, 0].set_title("Active tension and phase")
    ax[1, 1].plot(t, arr[:, 13], label="J min")
    ax[1, 1].plot(t, arr[:, 14], label="J max")
    ax[1, 1].set_title("det F")
    ax[1, 2].plot(t, arr[:, 2], label="lambda min")
    ax[1, 2].plot(t, arr[:, 3], label="lambda max")
    ax[1, 2].set_title("Fibre stretch, myocardium")
    for a in ax.flat:
        a.legend(loc="best")
        a.set_xlabel("t (ms)")
    ax[0, 2].set_xlabel("volume (mL)")

    if has_ecg:
        te, ve = np.asarray(ecg_t), np.asarray(ecg_v)
        col = {name: i for i, name in enumerate(ecg_leads)}
        t_hi = max(te[-1], t[-1])
        for r, row in enumerate(LEAD_ROWS):
            for c, lead in enumerate(row):
                a = fig.add_subplot(gs[2 + r, 2 * c:2 * c + 2])
                a.plot(te, ve[:, col[lead]], "k", linewidth=0.8)
                a.axhline(0.0, color="0.7", linewidth=0.5)
                a.set_xlim(te[0], t_hi)
                a.set_title(lead, fontsize=9, fontweight="bold")
                a.tick_params(labelsize=7)
                a.grid(True, alpha=0.3)
                if r == 1:
                    a.set_xlabel("t (ms)", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def load_ecg_csv(path):
    path = Path(path)
    if not path.exists():
        return None, None, None
    with open(path) as fh:
        leads = fh.readline().strip().split(",")[1:]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], data[:, 1:], leads