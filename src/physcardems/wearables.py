"""wearables.py: wearable-signal observation operators for physcardems runs.

One simulated beat (aortic outflow, arterial pressure and cycle phases from log.csv) is turned
into the signals a wearable would record:

  * PPG at the finger and wrist. The simulated aortic outflow drives a linear 1D arterial
    network (frequency domain, Womersley viscous losses, terminal 3-element Windkessels). As in
    the Pulse Wave Database (Charlton et al. 2019), the PPG is taken as blood volume: in the
    digital terminal Windkessel for the finger, and in the radial artery for the wrist.
  * Thoracic bioimpedance Z(t) and the impedance cardiogram -dZ/dt, from the two mechanisms in
    Badeli et al. 2020: distension of the thoracic aorta, and flow-dependent blood conductivity
    from red blood cell alignment.
  * Timing features: pulse arrival and transit times from the PPG, PEP and LVET from the
    impedance cardiogram, and the model's own PEP and LVET from its cycle phases, so each
    operator can be checked against the quantity it is meant to measure.

This is a demonstration operator. The network geometry and the impedance constants are
approximate literature-style values (NETWORK, BioZParams). Read waveform shape and timing,
not absolute PPG or impedance amplitudes. No dolfinx import: numpy and scipy only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import jve

RHO = 1050.0        # blood density (kg/m^3)
MU = 0.0035         # blood viscosity (Pa s)
MMHG = 133.322      # Pa per mmHg
WALL_LOSS = 0.1     # wall viscoelastic loss angle (rad); damps the higher harmonics
F_MAX_HZ = 40.0     # harmonics above this are dropped
IVC, EJECTION, IVR = 1, 2, 3   # phase codes in log.csv

# name: (parent, length cm, radius cm, share of cardiac output for terminal beds, else None)
# A reduced network: the path to the right hand in some detail and the other beds lumped, so
# the wrist and finger waves carry reflections from the rest of the tree. Approximate adult values.
NETWORK = {
    "asc_aorta":    (None,            4.0, 1.47, None),
    "arch_a":       ("asc_aorta",     2.0, 1.26, None),
    "brachioceph":  ("arch_a",        3.4, 0.62, None),
    "arch_b":       ("arch_a",        3.9, 1.12, None),
    "r_subclavian": ("brachioceph",   6.8, 0.42, None),
    "r_carotid":    ("brachioceph",  17.7, 0.37, 0.07),
    "l_carotid":    ("arch_b",       20.8, 0.37, 0.07),
    "l_arm":        ("arch_b",       40.0, 0.35, 0.04),
    "thor_aorta":   ("arch_b",       15.6, 1.00, None),
    "abd_aorta":    ("thor_aorta",   15.0, 0.75, None),
    "visceral":     ("abd_aorta",     3.0, 0.40, 0.45),
    "iliac":        ("abd_aorta",     5.8, 0.52, None),
    "leg":          ("iliac",        40.0, 0.37, 0.33),
    "r_brachial":   ("r_subclavian", 42.2, 0.31, None),
    "r_radial":     ("r_brachial",   23.5, 0.17, None),
    "r_ulnar":      ("r_brachial",   23.5, 0.20, 0.02),
    "digital":      ("r_radial",      8.0, 0.10, 0.02),
}
THORACIC = ("asc_aorta", "arch_a", "arch_b", "thor_aorta")   # vessels inside the ICG field


def wave_speed(r_m, stiffness_scale=1.0):
    """Moens-Korteweg wave speed (m/s), with the empirical wall stiffness Eh/r0 of Olufsen (1999)."""
    r_cm = r_m * 100.0
    eh_r0 = 0.1 * (2.0e6 * np.exp(-22.53 * r_cm) + 8.65e5)      # dyn/cm^2 -> Pa
    return np.sqrt(stiffness_scale * 2.0 / 3.0 * eh_r0 / RHO)


@dataclass
class Segment:
    name: str
    parent: str | None
    L: float                     # length (m)
    r: float                     # radius (m)
    share: float | None          # share of cardiac output (terminal beds only)
    stiffness_scale: float = 1.0
    children: list = field(default_factory=list)
    R1: float = 0.0              # terminal Windkessel: characteristic, peripheral resistance, compliance
    R2: float = 0.0
    C: float = 0.0

    def __post_init__(self):
        self.A0 = np.pi * self.r ** 2
        self.c0 = wave_speed(self.r, self.stiffness_scale)
        self.Cp = self.A0 / (RHO * self.c0 ** 2)                  # area compliance per length (m^2/Pa)
        self.R_pois = 8.0 * MU * self.L / (np.pi * self.r ** 4)   # steady-flow resistance

    @property
    def terminal(self):
        return self.share is not None

    def line(self, w):
        """Characteristic impedance and propagation constant at angular frequency w > 0."""
        z = self.r * np.sqrt(w * RHO / MU) * np.exp(0.75j * np.pi)
        f10 = 2.0 * jve(1, z) / (z * jve(0, z))                  # Womersley; jve keeps it finite
        c = self.c0 * np.sqrt((1.0 - f10) * np.exp(1j * WALL_LOSS))
        return RHO * c / (self.A0 * (1.0 - f10)), 1j * w / c


class ArterialNetwork:
    """Linear 1D arterial network, solved harmonic by harmonic for a periodic aortic inflow."""

    def __init__(self, R_total, C_total, stiffness_scale=1.0, network=NETWORK):
        self.seg = {name: Segment(name, parent, L_cm / 100.0, r_cm / 100.0, share, stiffness_scale)
                    for name, (parent, L_cm, r_cm, share) in network.items()}
        for s in self.seg.values():
            if s.parent is not None:
                self.seg[s.parent].children.append(s.name)
        self.root = next(name for name, s in self.seg.items() if s.parent is None)
        # peripheral compliance = total arterial compliance minus what the conduits hold
        c_conduit = sum(s.Cp * s.L for s in self.seg.values())
        c_per = C_total - c_conduit
        self.notes = []
        if c_per < 0.2 * C_total:
            self.notes.append(f"conduits hold {c_conduit / C_total:.0%} of the total compliance; "
                              "peripheral compliance set to 20% of the total")
            c_per = 0.2 * C_total
        for s in self.seg.values():
            if s.terminal:
                RT = R_total / s.share                   # beds in parallel share the total resistance
                s.R1 = RHO * s.c0 / s.A0                 # matched to the vessel, few spurious reflections
                s.R2 = max(RT - s.R1, 0.1 * RT)
                s.C = c_per * s.share

    def _impedances(self, name, w, k, res):
        s = self.seg[name]
        if s.terminal:
            zc = s.R2 / (1.0 + 1j * w * s.R2 * s.C)
            zl = s.R1 + zc
            res[name]["cratio"][k] = zc / zl             # compliance pressure / bed inlet pressure
        else:
            zl = 1.0 / sum(1.0 / self._impedances(ch, w, k, res) for ch in s.children)
        if w == 0.0:
            zin = s.R_pois + zl
            h = zl / zin
        else:
            z0, g = s.line(w)
            gl = g * s.L
            th = np.tanh(gl)
            zin = z0 * (zl + z0 * th) / (z0 + zl * th)
            h = 1.0 / (np.cosh(gl) + z0 / zl * np.sinh(gl))   # distal / proximal pressure
        res[name]["zin"][k], res[name]["h"][k] = zin, h
        return zin

    def _pressures(self, name, p_in, k, res):
        s, r = self.seg[name], res[name]
        r["p_in"][k] = p_in
        r["q_in"][k] = p_in / r["zin"][k]
        r["p_out"][k] = p_out = r["h"][k] * p_in
        if s.terminal:
            r["p_c"][k] = r["cratio"][k] * p_out
        for ch in s.children:
            self._pressures(ch, p_out, k, res)

    def solve(self, q, period_s):
        """Inflow q (m^3/s, one period, uniform samples) -> for every segment the proximal and
        distal pressure p_in, p_out (Pa), proximal flow q_in (m^3/s) and, for terminal beds, the
        Windkessel compliance pressure p_c (Pa). All returned as arrays like q."""
        n = len(q)
        q_hat = np.fft.rfft(q)
        w = 2.0 * np.pi * np.arange(len(q_hat)) / period_s
        keys = ("zin", "h", "cratio", "p_in", "p_out", "q_in", "p_c")
        res = {name: {k: np.zeros(len(q_hat), complex) for k in keys} for name in self.seg}
        for k in range(len(q_hat)):
            if w[k] > 2.0 * np.pi * F_MAX_HZ:
                break
            zin = self._impedances(self.root, w[k], k, res)
            self._pressures(self.root, zin * q_hat[k], k, res)
        return {name: {k: np.fft.irfft(r[k], n) for k in ("p_in", "p_out", "q_in", "p_c")}
                for name, r in res.items()}


# ---------------------------------------------------------------------------- bioimpedance
@dataclass
class BioZParams:
    """Thoracic impedance constants: approximate, chosen to give typical ICG magnitudes."""
    Z0_ohm: float = 25.0          # baseline thoracic impedance
    blood_share: float = 0.05     # share of thoracic conductance carried by aortic blood at rest
    k_rbc: float = 0.12           # rise in blood conductivity at full red cell alignment
    shear0_per_s: float = 50.0    # reduced shear rate |v|/r giving 63% alignment in steady flow
    tau_align_s: float = 0.01     # alignment time constant (fast)
    tau_relax_s: float = 0.10     # return to random orientation (slower)


def rbc_alignment(shear, dt, prm, settle_beats=3):
    """Red cell alignment state (0 random, 1 aligned), first order with faster alignment than
    relaxation; the beat is repeated until the state is periodic."""
    s_eq = 1.0 - np.exp(-shear / prm.shear0_per_s)
    s, out = 0.0, np.empty_like(s_eq)
    for _ in range(settle_beats):
        for i, se in enumerate(s_eq):
            a = dt / (prm.tau_align_s if se > s else prm.tau_relax_s)
            s = (s + a * se) / (1.0 + a)             # implicit Euler
            out[i] = s
    return out


def thoracic_impedance(net, flows, dt, prm=BioZParams()):
    """Z(t) (ohm): aortic blood in parallel with the other thoracic tissue. Its conductance
    changes with the blood volume (distension) and the blood conductivity (alignment)."""
    g, v_rest = 0.0, 0.0
    for name in THORACIC:
        s, f = net.seg[name], flows[name]
        p = 0.5 * (f["p_in"] + f["p_out"])
        area = s.A0 + s.Cp * (p - p.mean())
        shear = np.abs(f["q_in"] / area) / s.r
        sigma = 1.0 + prm.k_rbc * rbc_alignment(shear, dt, prm)
        g = g + sigma * area * s.L
        v_rest += s.A0 * s.L
    return prm.Z0_ohm / ((1.0 - prm.blood_share) + prm.blood_share * g / v_rest)


# ---------------------------------------------------------------------------- features
def pulse_foot(t, y):
    """Pulse foot by the intersecting-tangent method: the tangent at the steepest upstroke
    meets the horizontal through the preceding minimum."""
    dy = np.gradient(y, t)
    i_up = int(np.argmax(dy))
    i_min = int(np.argmin(y[: i_up + 1]))
    return t[i_up] - (y[i_up] - y[i_min]) / dy[i_up]


def icg_points(t, mdzdt):
    """B (valve opening), C (peak ejection) and X (valve closure) on -dZ/dt."""
    dt = t[1] - t[0]
    iC = int(np.argmax(mdzdt))
    i0 = max(0, iC - int(0.25 / dt))
    seg = mdzdt[i0: iC + 1]
    base = seg.min()
    iB = i0 + int(np.where(seg <= base + 0.1 * (mdzdt[iC] - base))[0][-1])
    iX = iC + int(np.argmin(mdzdt[iC: iC + int(0.4 / dt)]))
    return t[iB], t[iC], t[iX]


# ---------------------------------------------------------------------------- beats
def load_run(run_dir):
    run_dir = Path(run_dir)
    log = np.genfromtxt(run_dir / "log.csv", delimiter=",", names=True)
    cfg_path = run_dir / "config_used.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return log, cfg


def _col(log, *names):
    for name in names:
        if name in log.dtype.names:
            return np.asarray(log[name], float)
    raise KeyError(f"log.csv has none of the columns {names}")


def beat_from_log(log, pcl_ms, lat0_ms, n=1024):
    """The last beat that has reached IVR, as one period starting at its first activation
    (lat0_ms + a whole number of cycles). Aortic flow is taken during ejection only."""
    t = _col(log, "t_ms")

    ph = _col(log, "phase", "phase_lv").astype(int)
    q_log = np.clip(_col(log, "Q_mL_s", "Q_lv_mL_s"), 0.0, None)
    if not np.any(q_log[ph == EJECTION] > 0):          # outflow not logged: use -dV/dt
        q_log = np.clip(-np.gradient(_col(log, "V_mL", "V_lv_mL"), t / 1e3), 0.0, None)
    q = np.where(ph == EJECTION, q_log, 0.0) * 1e-6
    p_lv = _col(log, "P_kPa", "P_lv_kPa") * 1e3
    p_art = _col(log, "P_art_kPa", "Pc_lv_kPa", "Part_lv_kPa") * 1e3

    def entries(code):
        return np.where((ph[1:] == code) & (ph[:-1] != code))[0] + 1

    ivc, ivr = entries(IVC), entries(IVR)
    done = [i for i in ivc if np.any(ivr > i)]
    if not done:
        raise ValueError("no beat has reached isovolumic relaxation yet")
    i_ivc = done[-1]
    i_ivr = ivr[ivr > i_ivc][0]
    i_open = i_ivc + int(np.argmax(ph[i_ivc:] == EJECTION))
    # t_act = lat0_ms + np.floor((t[i_ivc] - lat0_ms) / pcl_ms) * pcl_ms
    # anchor on valve opening: the timed IVC entry can come just before the first activation
    t_act = lat0_ms + np.floor((t[i_open] - lat0_ms) / pcl_ms) * pcl_ms
    tt = t_act + np.arange(n) * pcl_ms / n

    def window(x):
        return np.interp(tt, t, x, left=np.nan, right=np.nan)

    if not np.any(np.nan_to_num(window(q)) > 0):
        raise ValueError(f"no ejection between t = {t_act:.0f} and {t_act + pcl_ms:.0f} ms; "
                         f"check --lat0 (valve opened at t = {t[i_open]:.0f} ms)")

    return dict(t_s=(tt - t_act) / 1e3, q=np.nan_to_num(window(q)),   # valve shut outside the log
                p_lv=window(p_lv), p_art=window(p_art), period_s=pcl_ms / 1e3,
                t_open_s=(t[i_open] - t_act) / 1e3, t_close_s=(t[i_ivr] - t_act) / 1e3,
                t_act_ms=t_act)


def synthetic_beat(pcl_ms=800.0, sv_ml=65.0, pep_ms=70.0, lvet_ms=290.0, map_mmhg=85.0, n=1024):
    """Stand-in beat with a skewed half-sine aortic flow, to test the operators without a run."""
    t = np.arange(n) * pcl_ms / n / 1e3
    x = (t - pep_ms / 1e3) / (lvet_ms / 1e3)
    shape = np.where((x > 0) & (x < 1), np.sin(np.pi * x) * np.exp(-1.5 * x), 0.0)
    q = shape * sv_ml * 1e-6 / (shape.sum() * (t[1] - t[0]))
    return dict(t_s=t, q=q, p_lv=np.full(n, np.nan), p_art=np.full(n, map_mmhg * MMHG),
                period_s=pcl_ms / 1e3, t_open_s=pep_ms / 1e3, t_close_s=(pep_ms + lvet_ms) / 1e3)


# ---------------------------------------------------------------------------- all together
def wearable_signals(beat, C_total_ml_per_mmhg=1.5, stiffness_scale=1.0, bioz=BioZParams(),
                     R_fallback_mmhg_s_per_ml=1.0):
    """PPG, impedance and timing features for one beat (dict from beat_from_log or synthetic_beat)."""
    t, q = beat["t_s"], beat["q"]
    map_pa = np.nanmean(beat["p_art"]) if np.isfinite(beat["p_art"]).any() else np.nan
    R_total = map_pa / q.mean() if np.isfinite(map_pa) else R_fallback_mmhg_s_per_ml * MMHG * 1e6
    net = ArterialNetwork(R_total, C_total_ml_per_mmhg * 1e-6 / MMHG, stiffness_scale)
    flows = net.solve(q, beat["period_s"])

    p_root = flows[net.root]["p_in"]
    finger = flows["digital"]["p_c"] * net.seg["digital"].C                     # blood volume, finger bed
    rad = net.seg["r_radial"]
    wrist = 0.5 * (flows["r_radial"]["p_in"] + flows["r_radial"]["p_out"]) * rad.Cp * rad.L
    z = thoracic_impedance(net, flows, t[1] - t[0], bioz)
    mdzdt = -np.gradient(z, t)

    tB, tC, tX = icg_points(t, mdzdt)
    f_root, f_finger, f_wrist = (pulse_foot(t, y) for y in (p_root, finger, wrist))
    ms = 1e3
    features = {
        "PAT_finger_ms": f_finger * ms,            # first activation (Q onset) to finger PPG foot
        "PAT_wrist_ms": f_wrist * ms,
        "PTT_finger_ms": (f_finger - f_root) * ms,  # aortic root foot to finger PPG foot
        "PEP_icg_ms": tB * ms,                      # Q onset to B point
        "LVET_icg_ms": (tX - tB) * ms,              # B to X
        "PEP_model_ms": beat["t_open_s"] * ms,      # ground truth from the cycle phases
        "LVET_model_ms": (beat["t_close_s"] - beat["t_open_s"]) * ms,
        "finger_rise_ms": (t[int(np.argmax(finger))] - f_finger) * ms,
        "dZdt_max_ohm_per_s": float(mdzdt.max()),
        "deltaZ_ohm": float(z.max() - z.min()),
        "aortic_SBP_DBP_mmHg": [p_root.max() / MMHG, p_root.min() / MMHG],
        "radial_SBP_DBP_mmHg": [flows["r_radial"]["p_out"].max() / MMHG,
                                flows["r_radial"]["p_out"].min() / MMHG],
    }
    return dict(net=net, flows=flows, p_root=p_root, finger=finger, wrist=wrist, z=z,
                mdzdt=mdzdt, icg=(tB, tC, tX), feet=(f_root, f_finger, f_wrist),
                features=features, notes=net.notes)