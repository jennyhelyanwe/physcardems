"""pseudo_ecg.py

dolfinx port of the legacy simcardems pseudo-ECG (simulations/rodero_05/pseudo_ecg.py
and simcardems.postprocess.ecg_recovery):

    phi(x_e) = 1/(4 pi sigma_b) * integral( G grad(v) . r / |r|^3 dx ),  r = x - x_e

G is the identity (legacy) or a conductivity tensor. The integral is independent of the
mesh length unit, provided electrode coordinates use the same unit as the mesh.
"""
import numpy as np
import dolfinx
import ufl
from mpi4py import MPI

ELECTRODE_ORDER = ("LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6")
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
LEAD_LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]


def load_electrodes(path, unit_scale):
    """Rows LA, RA, LL, RL, V1..V6 (as in the Alya electrode file), scaled into mesh units."""
    xyz = np.loadtxt(path, delimiter=",") * unit_scale
    assert xyz.shape == (10, 3), f"expected 10 electrodes, got {xyz.shape}"
    return dict(zip(ELECTRODE_ORDER, xyz))


def standard_leads(electrodes):
    """Leads as linear combinations of electrode potentials (Einthoven, Goldberger, Wilson)."""
    points = {k: electrodes[k] for k in ("LA", "RA", "LL", *PRECORDIAL)}
    leads = {
        "I": {"LA": 1.0, "RA": -1.0},
        "II": {"LL": 1.0, "RA": -1.0},
        "III": {"LL": 1.0, "LA": -1.0},
        "aVR": {"RA": 1.0, "LA": -0.5, "LL": -0.5},
        "aVL": {"LA": 1.0, "RA": -0.5, "LL": -0.5},
        "aVF": {"LL": 1.0, "RA": -0.5, "LA": -0.5},
    }
    for v in PRECORDIAL:
        leads[v] = {v: 1.0, "LA": -1.0 / 3.0, "RA": -1.0 / 3.0, "LL": -1.0 / 3.0}
    return points, leads


def legacy_leads(electrodes):
    """Legacy definitions: reference potentials evaluated at averaged electrode positions."""
    LA, RA, LL = electrodes["LA"], electrodes["RA"], electrodes["LL"]
    points = {k: electrodes[k] for k in ("LA", "RA", "LL", *PRECORDIAL)}
    points.update({
        "WCT_pt": (LA + RA + LL) / 3.0,
        "LA_LL_mid": (LA + LL) / 2.0,
        "RA_LL_mid": (RA + LL) / 2.0,
        "RA_LA_mid": (RA + LA) / 2.0,
    })
    leads = {
        "I": {"LA": 1.0, "RA": -1.0},
        "II": {"LL": 1.0, "RA": -1.0},
        "III": {"LL": 1.0, "LA": -1.0},
        "aVR": {"RA": 1.0, "LA_LL_mid": -1.0},
        "aVL": {"LA": 1.0, "RA_LL_mid": -1.0},
        "aVF": {"LL": 1.0, "RA_LA_mid": -1.0},
    }
    for v in PRECORDIAL:
        leads[v] = {v: 1.0, "WCT_pt": -1.0}
    return points, leads


class PseudoECG:
    def __init__(self, v, points, leads, sigma_b=1.0, G=None, quadrature_degree=2):
        mesh = v.function_space.mesh
        self.comm = mesh.comm
        x = ufl.SpatialCoordinate(mesh)
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
        flux = ufl.grad(v) if G is None else G * ufl.grad(v)
        self.forms = {}
        for name, p in points.items():
            xe = dolfinx.fem.Constant(mesh, np.asarray(p, dtype=dolfinx.default_scalar_type))
            r = x - xe
            self.forms[name] = dolfinx.fem.form(ufl.inner(flux, r) / ufl.dot(r, r) ** 1.5 * dx)
        self.scale = 1.0 / (4.0 * np.pi * sigma_b)
        self.leads = leads
        self.times = []
        self.values = []

    def potentials(self):
        return {
            name: self.scale * self.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
            for name, form in self.forms.items()
        }

    def compute(self, t):
        phi = self.potentials()
        row = [sum(c * phi[p] for p, c in comb.items()) for comb in self.leads.values()]
        self.times.append(t)
        self.values.append(row)
        return dict(zip(self.leads, row))

    def save_csv(self, path):
        if self.comm.rank != 0 or not self.times:
            return
        data = np.column_stack([self.times, np.array(self.values)])
        np.savetxt(path, data, delimiter=",", header="t_ms," + ",".join(self.leads), comments="", fmt="%.8g")

    def plot(self, path, t_window=None):
        if self.comm.rank != 0 or not self.times:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array(self.times)
        vals = np.array(self.values)
        idx = {name: i for i, name in enumerate(self.leads)}
        fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
        for r, row in enumerate(LEAD_LAYOUT):
            for c, lead in enumerate(row):
                ax = axes[r, c]
                ax.plot(t, vals[:, idx[lead]], "k", lw=1)
                ax.axhline(0.0, color="0.7", lw=0.5)
                ax.set_title(lead, fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.3)
                if t_window is not None:
                    ax.set_xlim(*t_window)
                if r == 2:
                    ax.set_xlabel("t (ms)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)