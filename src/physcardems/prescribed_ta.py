"""prescribed.py: active tension prescribed as an analytical profile of time since activation.

Ta(x, t) = Ta_d + (Ta_peak - Ta_d) * shape(t - LAT(x)), with per-cell-type shape parameters.
shape (tau = time since activation, ms, wrapped by the cycle length):
    rise  (0 <= tau < t_peak):   sin^2( pi/2 * (tau / t_peak)^a_rise )
    decay (t_peak <= tau <= pcl): Gaussian of width sigma_decay, rescaled to reach 0 at tau = pcl
so Ta = Ta_d at activation and at the end of the cycle, and Ta = Ta_peak at t_peak.
Before the first activation Ta = Ta_d.

No length or velocity dependence: the sweep finds the tension the tissue must actually develop.
"""
from __future__ import annotations

import numpy as np
import dolfinx
import ufl
import pulse

CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}
SHAPE_KEYS = ("Ta_peak_kpa", "t_peak_ms", "a_rise", "sigma_decay_ms")
REL_KEYS = {"Ta_peak_kpa": "Ta_peak", "t_peak_ms": "t_peak", "a_rise": "a_rise", "sigma_decay_ms": "sigma_decay"}


def ta_shape(tau, t_peak, a_rise, sigma_decay, pcl):
    """Normalised shape in [0, 1]; tau is time since activation (ms), already wrapped to [0, pcl)."""
    tau = np.asarray(tau, dtype=float)
    s = np.clip(tau / t_peak, 0.0, 1.0)
    rise = np.sin(0.5 * np.pi * s**a_rise) ** 2
    e1 = np.exp(-0.5 * ((pcl - t_peak) / sigma_decay) ** 2)
    fall = (np.exp(-0.5 * ((tau - t_peak) / sigma_decay) ** 2) - e1) / (1.0 - e1)
    return np.where(tau < t_peak, rise, np.clip(fall, 0.0, 1.0))


def celltype_profiles(active_cfg, pcl_ms):
    """Per-cell-type shape parameters from the [active] config section.

    endo values are given directly; epi and mid are endo values times the ratios in
    [active.relative.epi] and [active.relative.mid] (keys as in endo, without units).
    """
    endo = {k: float(active_cfg["endo"][k]) for k in SHAPE_KEYS}
    out = {"endo": endo}
    for name in ("epi", "mid"):
        rel = dict(active_cfg.get("relative", {}).get(name, {}))
        unknown = set(rel) - set(REL_KEYS.values())
        assert not unknown, f"[active.relative.{name}]: unknown keys {unknown}; use {sorted(REL_KEYS.values())}"
        out[name] = {k: endo[k] * float(rel.get(r, 1.0)) for k, r in REL_KEYS.items()}
    for name, p in out.items():
        assert 0.0 < p["t_peak_ms"] < pcl_ms, f"{name}: t_peak outside the cycle"
        assert p["a_rise"] > 0.0 and p["sigma_decay_ms"] > 0.0, f"{name}: a_rise and sigma_decay must be > 0"
    return out


def ta_nodal(t_ms, lat_ms, celltype, profiles, ta_d_kpa, pcl_ms):
    """Ta in kPa at each node at time t_ms."""
    tau = t_ms - lat_ms
    active = tau >= 0.0
    tau_w = np.mod(np.where(active, tau, 0.0), pcl_ms)
    ta = np.full(lat_ms.shape, ta_d_kpa, dtype=float)
    for c, name in CELLTYPES.items():
        sel = active & (celltype == c)
        if not np.any(sel):
            continue
        p = profiles[name]
        shape = ta_shape(tau_w[sel], p["t_peak_ms"], p["a_rise"], p["sigma_decay_ms"], pcl_ms)
        ta[sel] = ta_d_kpa + (p["Ta_peak_kpa"] - ta_d_kpa) * shape
    return ta


class PrescribedActive:
    """Active stress along f0 with Ta taken from a nodal (P1) field updated each step.

    Interface used by run_mech_ta.py and ControlledCavityDynamicProblem: register(u),
    set_time(t_ms), S(C, dev), P(F, dev), post_solve(), lmbda, Ta_current (Pa, DG1).
    """

    def __init__(self, *, f0, mesh, lat_ms, celltype, profiles, ta_d_kpa, pcl_ms,
                 formulation=pulse.ActiveStressFormulation.stretch):
        self.f0 = f0
        self.formulation = formulation
        self.lat_ms = np.asarray(lat_ms, dtype=float)
        self.celltype = np.asarray(celltype, dtype=int)
        self.profiles = profiles
        self.ta_d_kpa = float(ta_d_kpa)
        self.pcl_ms = float(pcl_ms)

        P1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        assert P1.dofmap.index_map.size_local + P1.dofmap.index_map.num_ghosts == self.lat_ms.size, \
            "LAT and cell-type arrays must be P1 nodal arrays on the mechanics mesh"
        self.Ta = dolfinx.fem.Function(P1, name="Ta_Pa")          # enters the form
        self.function_space = dolfinx.fem.functionspace(mesh, ("DG", 1))
        self.lmbda = dolfinx.fem.Function(self.function_space, name="lambda")
        self.Ta_current = dolfinx.fem.Function(self.function_space, name="Ta")
        self.u = None
        self.set_time(0.0)

    def register(self, u):
        self.u = u

    def set_time(self, t_ms):
        self.Ta.x.array[:] = 1e3 * ta_nodal(t_ms, self.lat_ms, self.celltype, self.profiles,
                                            self.ta_d_kpa, self.pcl_ms)

    def strain_energy(self, C):
        raise NotImplementedError("PrescribedActive supplies S directly")

    def S(self, C, dev=False):
        lmbda = ufl.sqrt(ufl.inner(C * self.f0, self.f0))
        Ta = self.Ta / lmbda if self.formulation == pulse.ActiveStressFormulation.stretch else self.Ta
        return Ta * ufl.outer(self.f0, self.f0)

    def P(self, F, dev=False):
        return F * self.S(F.T * F, dev=dev)

    def post_solve(self):
        F = ufl.grad(self.u) + ufl.Identity(3)
        lmbda = ufl.sqrt(ufl.inner(F.T * F * self.f0, self.f0))
        pts = self.function_space.element.interpolation_points
        self.lmbda.interpolate(dolfinx.fem.Expression(lmbda, pts))
        self.Ta_current.interpolate(self.Ta)