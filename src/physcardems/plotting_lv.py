"""plotting_lv.py: log format, beat-by-beat biomarkers and the run summary figure for
LV-only simulations (run_em_lvonly.py, replot_summary_lv.py).

log.csv has one row per mechanics step, with the columns in LOG_COLUMNS. Read it with
load_log(), which returns a dict of numpy arrays keyed by column name, e.g. log["V_mL"].

Phases (column "phase"): 0 PRELOAD, 1 IVC, 2 EJECTION, 3 IVR, 4 FILLING.
A beat starts at entry to IVC. Its biomarkers are final once it has entered IVR.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# same as physcardems.cycle.PHASE_NAMES; copied so this module imports without dolfinx
PHASE_NAMES = {0: "PRELOAD", 1: "IVC", 2: "EJECTION", 3: "IVR", 4: "FILLING"}

LOG_COLUMNS = (
    "t_ms",          # time
    "Ta_max_kPa",    # maximum active tension in the myocardium
    "lambda_min",    # fibre stretch range in the myocardium
    "lambda_max",
    "phase",         # cardiac cycle phase (PHASE_NAMES)
    "V_mL",          # LV cavity volume
    "P_kPa",         # LV cavity pressure
    "P_art_kPa",     # arterial (Windkessel compliance) pressure
    "Q_mL_s",        # LV outflow, positive during ejection
    "newton_its",    # Newton iterations for the mechanics step
    "J_min",         # range of det F (volume change); J <= 0 means an inverted element
    "J_max",
    "n_inverted",    # number of inverted elements
    "v_min_mV",      # transmembrane potential range
    "v_max_mV",
    "wall_ep_s",     # wall-clock time for the EP substeps
    "wall_s",        # wall-clock time for the whole step
)
LOG_HEADER = ",".join(LOG_COLUMNS)
LEAD_ROWS = [["I", "II", "III", "aVR", "aVL", "aVF"], ["V1", "V2", "V3", "V4", "V5", "V6"]]


def log_from_rows(rows) -> dict:
    """List of rows (in LOG_COLUMNS order) -> dict of arrays."""
    arr = np.atleast_2d(np.asarray(rows, dtype=float))
    return {name: arr[:, i] for i, name in enumerate(LOG_COLUMNS)}


def save_log(log: dict, path) -> None:
    arr = np.column_stack([log[name] for name in LOG_COLUMNS])
    np.savetxt(path, arr, delimiter=",", header=LOG_HEADER, comments="")


def load_log(path) -> dict:
    with open(path) as fh:
        names = fh.readline().strip().split(",")
    assert tuple(names) == LOG_COLUMNS, f"{path} is not an LV-only log (columns {names})"
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {name: arr[:, i] for i, name in enumerate(names)}


def beat_biomarkers(log: dict) -> list[dict]:
    """Biomarkers for each beat, a beat running from one entry to IVC to the next.

    EDV and EDP at entry to IVC; ESV at entry to IVR; Pmax over the beat; peak ejection
    rate (mL/s) over ejection; peak dP/dt (kPa/s) over the beat. 'complete' is True once the
    beat has entered IVR. Before that, ESV is the minimum volume so far.
    """
    t, V, P, ph = log["t_ms"], log["V_mL"], log["P_kPa"], log["phase"]
    n = len(t)
    if n < 2:
        return []
    starts = np.flatnonzero((ph == 1) & (np.r_[-1, ph[:-1]] != 1))
    dvdt = -np.gradient(V, t) * 1e3      # mL/s, positive when the volume falls
    dpdt = np.gradient(P, t) * 1e3       # kPa/s
    beats = []
    for k, i0 in enumerate(starts):
        i1 = starts[k + 1] if k + 1 < len(starts) else n
        seg = np.arange(i0, i1)
        ivr = seg[(ph[seg] == 3) & (np.r_[-1, ph[seg][:-1]] != 3)]
        complete = ivr.size > 0
        edv = V[i0]
        esv = V[ivr[0]] if complete else V[seg].min()
        ej = seg[ph[seg] == 2]
        beats.append({
            "beat": k + 1,
            "complete": bool(complete),
            "t_start_ms": float(t[i0]),
            "EDV_mL": float(edv),
            "ESV_mL": float(esv),
            "SV_mL": float(edv - esv),
            "LVEF_pct": float(100.0 * (edv - esv) / edv),
            "EDP_kPa": float(P[i0]),
            "Pmax_kPa": float(P[seg].max()),
            "dvdt_ejection_mL_s": float(dvdt[ej].max()) if ej.size else float("nan"),
            "dpdt_max_kPa_s": float(dpdt[seg].max()),
        })
    return beats


def last_complete_beat(log: dict) -> dict | None:
    done = [b for b in beat_biomarkers(log) if b["complete"]]
    return done[-1] if done else None


def _beat_text(b):
    tag = "" if b["complete"] else " (ejecting)"
    return (f"beat {b['beat']}{tag}: EF {b['LVEF_pct']:.1f}%, EDV {b['EDV_mL']:.1f}, "
            f"ESV {b['ESV_mL']:.1f}, SV {b['SV_mL']:.1f} mL\n"
            f"EDP {b['EDP_kPa']:.2f} kPa, Pmax {b['Pmax_kPa']:.1f} kPa, "
            f"peak ejection {b['dvdt_ejection_mL_s']:.0f} mL/s, dP/dt max {b['dpdt_max_kPa_s']:.0f} kPa/s")


PHASE_SHADING = {1: ("IVC", "#fde0c5"), 2: ("ejection", "#f9b98e"), 3: ("IVR", "#d9d2e9")}


def shade_phases(ax, t, phase, legend=False):
    """Shade IVC, ejection and IVR as background bands."""
    edges = np.flatnonzero(np.diff(phase) != 0) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(t)]
    labelled = set()
    for a, b in zip(starts, ends):
        p = int(phase[a])
        if p in PHASE_SHADING:
            name, colour = PHASE_SHADING[p]
            t0 = t[a - 1] if a > 0 else t[a]
            ax.axvspan(t0, t[b - 1], color=colour, lw=0, zorder=0,
                       label=name if legend and p not in labelled else None)
            labelled.add(p)


def plot_summary(log: dict, path, title="", ecg_t=None, ecg_v=None, ecg_leads=None):
    """Mechanics panels (phases shaded: IVC, ejection, IVR), then the 12-lead pseudo-ECG if given."""
    t, ph = log["t_ms"], log["phase"]
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 12, height_ratios=[1.0, 1.0, 0.55, 0.55], hspace=0.6, wspace=1.0)
    top = [fig.add_subplot(gs[0, 4 * c:4 * c + 4]) for c in range(3)]
    mid = [fig.add_subplot(gs[1, 3 * c:3 * c + 3]) for c in range(4)]

    a = top[0]
    shade_phases(a, t, ph, legend=True)
    a.plot(t, log["P_kPa"], "C0", label="LV")
    a.plot(t, log["P_art_kPa"], "C3--", label="arterial (Windkessel)")
    a.set(title="Pressure (kPa)", xlabel="t (ms)")
    a.legend(loc="upper right", fontsize=8)

    a = top[1]
    shade_phases(a, t, ph)
    a.plot(t, log["V_mL"], "C0")
    a.set(title="LV volume (mL)", xlabel="t (ms)")

    beats = beat_biomarkers(log)
    a = top[2]
    a.plot(log["V_mL"], log["P_kPa"], "C0")
    a.set(xlabel="volume (mL)", ylabel="pressure (kPa)")
    a.set_title("PV loop\n" + (_beat_text(beats[-1]) if beats else "no IVC yet"), fontsize=10)

    panels = (
        (mid[0], "LV outflow (mL/s)", [("Q_mL_s", "C0-", None)]),
        (mid[1], "Max active tension, myocardium (kPa)", [("Ta_max_kPa", "k-", None)]),
        (mid[2], "det F (J)", [("J_min", "C0-", "min"), ("J_max", "C1-", "max")]),
        (mid[3], "Fibre stretch, myocardium", [("lambda_min", "C0-", "min"), ("lambda_max", "C1-", "max")]),
    )
    for a, name, curves in panels:
        shade_phases(a, t, ph)
        for col, style, lab in curves:
            a.plot(t, log[col], style, label=lab)
        a.set(title=name, xlabel="t (ms)")
        if any(lab for _, _, lab in curves):
            a.legend(loc="best", fontsize=8)

    if ecg_t is not None and len(ecg_t) > 1:
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

    if len(beats) > 1:
        title = f"{title}\n" + "; ".join(
            f"beat {b['beat']} EF {b['LVEF_pct']:.1f}%{'' if b['complete'] else ' (ejecting)'}" for b in beats)
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