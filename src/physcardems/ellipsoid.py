"""ellipsoid.py

Build LV ellipsoid cases for physcardems: geometry, fibres, EP node fields
(activation times, cell types) and ECG electrode positions.

Conventions (shared by the generator and the legacy converter)
  * units: metres
  * long axis along x, apex at negative x, basal plane at x = 0
    (pulse's cavity volume form -1/3 (x + u).n is then exact with the base held at x = 0)
  * +y is the septal direction (places the earliest activation sites and orients the
    heart in the torso for the ECG)
  * facet tags LV (endocardium), EPI, BASE, as in the BiV cases
  * fibres DG0, orthonormal f0, s0, n0
  * EP node fields on P1: lat_ms, celltype (ToR-ORd 0 endo, 1 epi, 2 mid), sf_IKs, transmural

The activation map is a graph eikonal (multi-source Dijkstra over first and second
neighbour mesh edges) with anisotropic myocardial conduction and a fast endocardial
layer standing in for the Purkinje network. It is a prescribed activation sequence
for teaching, not a calibrated conduction model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

CELLTYPE = {"endo": 0, "epi": 1, "mid": 2}  # ToR-ORd convention


# ---------------------------------------------------------------------------- parameters
@dataclass
class Anatomy:
    """Unloaded (reference) LV ellipsoid, in mm, truncated base_height_mm above the equator."""
    r_short_endo_mm: float = 21.0
    r_short_epi_mm: float = 30.0
    r_long_endo_mm: float = 68.0
    r_long_epi_mm: float = 76.0
    base_height_mm: float = 8.0
    mesh_size_mm: float = 6.0
    fibre_angle_endo_deg: float = -60.0   # cardiac-geometries convention; see helix_angles()
    fibre_angle_epi_deg: float = 60.0


@dataclass
class Activation:
    cv_fibre_m_per_s: float = 0.6
    cv_cross_m_per_s: float = 0.3
    cv_endo_m_per_s: float = 2.0                  # fast endocardial layer (Purkinje stand-in)
    endo_layer_max_height: float = 0.85           # Purkinje-covered endocardium, apex 0 -> base 1
    # earliest sites: (normalised height apex 0 -> base 1, circumferential angle from +y in deg)
    roots: list = field(default_factory=lambda: [[0.45, 0.0], [0.6, 90.0], [0.6, -90.0]])


@dataclass
class CellTypes:
    endo_fraction: float = 0.3    # transmural coordinate 0 (endo) to 1 (epi)
    mid_fraction: float = 0.3


def analytic_volumes_ml(a: Anatomy) -> dict:
    """Closed-form volumes of the truncated prolate ellipsoids (reference configuration)."""
    def trunc(rs, rl, h):
        return math.pi * rs**2 * ((h + rl) - (h**3 + rl**3) / (3 * rl**2)) * 1e-3
    v_endo = trunc(a.r_short_endo_mm, a.r_long_endo_mm, a.base_height_mm)
    v_epi = trunc(a.r_short_epi_mm, a.r_long_epi_mm, a.base_height_mm)
    return {"cavity_ml": v_endo, "wall_ml": v_epi - v_endo}


# ---------------------------------------------------------------------------- geometry
def generate_geometry(outdir: Path, anatomy: Anatomy, comm=None):
    """Mesh with cardiac-geometries (gmsh) with DG0 analytic fibres, then convert to
    metres with the basal plane at x = 0. Returns (Geometry, temporary mm folder)."""
    import cardiac_geometries as cg
    from mpi4py import MPI
    comm = comm or MPI.COMM_WORLD
    a = anatomy
    tmp = Path(outdir).with_name(Path(outdir).name + "_mm")
    geo = cg.mesh.lv_ellipsoid(
        tmp, r_short_endo=a.r_short_endo_mm, r_short_epi=a.r_short_epi_mm,
        r_long_endo=a.r_long_endo_mm, r_long_epi=a.r_long_epi_mm,
        mu_base_endo=-math.acos(a.base_height_mm / a.r_long_endo_mm),
        mu_base_epi=-math.acos(a.base_height_mm / a.r_long_epi_mm),
        psize_ref=a.mesh_size_mm, create_fibers=True,
        fiber_angle_endo=a.fibre_angle_endo_deg, fiber_angle_epi=a.fibre_angle_epi_deg,
        fiber_space="DG_0", comm=comm,
    )
    x = geo.mesh.geometry.x
    x[:, 0] -= a.base_height_mm
    x *= 1e-3
    return geo, tmp


def standardise_markers(markers: dict) -> dict:
    """cardiac-geometries ENDO/EPI/BASE -> physcardems LV/EPI/BASE (facets only)."""
    rename = {"ENDO": "LV", "EPI": "EPI", "BASE": "BASE"}
    return {rename[k]: tuple(int(i) for i in v) for k, v in markers.items() if k in rename}


def orthonormalise(f, s):
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    s = s - np.sum(s * f, axis=1, keepdims=True) * f
    s = s / np.linalg.norm(s, axis=1, keepdims=True)
    return f, s, np.cross(f, s)


# ---------------------------------------------------------------------------- EP node fields
def transmural_coordinate(mesh, ffun, endo_tag, epi_tag):
    """Laplace solution, 0 on the endocardium and 1 on the epicardium, on P1."""
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    fdim = mesh.topology.dim - 1
    bcs = []
    for tag, val in ((endo_tag, 0.0), (epi_tag, 1.0)):
        dofs = dolfinx.fem.locate_dofs_topological(V, fdim, ffun.find(tag))
        bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(val), dofs, V))
    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    problem = dolfinx.fem.petsc.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, zero * v * ufl.dx, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="transmural_")
    t = problem.solve()
    t.name = "transmural"
    return t


def celltypes_from_transmural(t, ct: CellTypes):
    out = np.full(t.shape, CELLTYPE["epi"], dtype=float)
    out[t < ct.endo_fraction + ct.mid_fraction] = CELLTYPE["mid"]
    out[t < ct.endo_fraction] = CELLTYPE["endo"]
    return out


def _edges_with_cells(tets):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    e = np.vstack([tets[:, [i, j]] for i, j in pairs])
    c = np.tile(np.arange(len(tets)), len(pairs))
    return np.sort(e, axis=1), c


def _anisotropic_edge_times(x, edges, f, cv_f, cv_t):
    d = x[edges[:, 1]] - x[edges[:, 0]]
    L2 = np.sum(d * d, axis=1)
    df = np.sum(d * f, axis=1)
    # t = sqrt(d^T D^-1 d), D = cv_t^2 I + (cv_f^2 - cv_t^2) f f^T
    return np.sqrt(np.maximum((L2 - df**2) / cv_t**2 + df**2 / cv_f**2, 0.0))


def activation_times(x, tets, cell_f0, endo_nodes, act: Activation):
    """Graph eikonal LAT (ms, starting at 0) at the vertices x (m)."""
    n = len(x)
    e1, c1 = _edges_with_cells(tets)
    # second-neighbour edges reduce the metrication error of graph distances
    A = sp.coo_matrix((np.ones(len(e1)), (e1[:, 0], e1[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    A2 = (A @ A).tocoo()
    m2 = A2.row < A2.col
    e2 = np.column_stack([A2.row[m2], A2.col[m2]])
    # fibre direction on second-neighbour edges: principal direction of the averaged nodal f f^T
    node_T = np.zeros((n, 3, 3))
    cnt = np.zeros(n)
    for k in range(4):
        np.add.at(node_T, tets[:, k], np.einsum("ci,cj->cij", cell_f0, cell_f0))
        np.add.at(cnt, tets[:, k], 1.0)
    node_T /= cnt[:, None, None]
    _, vv = np.linalg.eigh(node_T[e2[:, 0]] + node_T[e2[:, 1]])
    f2 = vv[:, :, -1]

    t1 = _anisotropic_edge_times(x, e1, cell_f0[c1], act.cv_fibre_m_per_s, act.cv_cross_m_per_s)
    t2 = _anisotropic_edge_times(x, e2, f2, act.cv_fibre_m_per_s, act.cv_cross_m_per_s)
    edges = np.vstack([e1, e2])
    times = np.concatenate([t1, t2]) * 1e3  # ms
    key = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    order = np.lexsort((times, key))       # duplicates: keep the fastest
    key, edges, times = key[order], edges[order], times[order]
    first = np.r_[True, key[1:] != key[:-1]]
    edges, times = edges[first], times[first]

    # endocardial layer times from the root sites, on the endocardial surface graph
    height = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    endo = np.asarray(endo_nodes)
    covered = endo[height[endo] <= act.endo_layer_max_height]
    is_cov = np.zeros(n, bool)
    is_cov[covered] = True
    ee = e1[is_cov[e1[:, 0]] & is_cov[e1[:, 1]]]
    L = np.linalg.norm(x[ee[:, 1]] - x[ee[:, 0]], axis=1) / act.cv_endo_m_per_s * 1e3
    G_endo = sp.coo_matrix((L, (ee[:, 0], ee[:, 1])), shape=(n, n)).tocsr()
    G_endo = G_endo.maximum(G_endo.T)
    r = x[covered, 1:] / np.maximum(np.linalg.norm(x[covered, 1:], axis=1, keepdims=True), 1e-12)
    roots = []
    for h, ang in act.roots:
        th = math.radians(ang)
        score = np.abs(height[covered] - h) * 4.0 + (1.0 - r @ np.array([math.cos(th), math.sin(th)]))
        roots.append(int(covered[np.argmin(score)]))
    t_endo = dijkstra(G_endo, indices=roots, min_only=True)

    # super-source n, linked to covered endocardial nodes with weight t_endo
    src = np.full(len(covered), n)
    rows = np.concatenate([edges[:, 0], edges[:, 1], src])
    cols = np.concatenate([edges[:, 1], edges[:, 0], covered])
    vals = np.concatenate([times, times, t_endo[covered] + 1e-6])
    G = sp.coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    lat = dijkstra(G, indices=n, directed=True)[:n]
    assert np.all(np.isfinite(lat)), "some nodes were not reached by the activation"
    return lat - lat.min(), np.array(roots)


# ---------------------------------------------------------------------------- ECG electrodes
# Body frame (x: patient left, y: anterior, z: superior), cm, relative to the LV centre.
BODY_ELECTRODES_CM = {
    "LA": (70.0, 0.0, 30.0), "RA": (-70.0, 0.0, 30.0), "LL": (15.0, 0.0, -80.0), "RL": (-15.0, 0.0, -80.0),
    "V1": (-7.0, 9.0, 3.0), "V2": (-2.0, 10.0, 3.0), "V3": (2.0, 10.0, -1.0),
    "V4": (6.0, 8.5, -4.0), "V5": (10.0, 5.0, -4.0), "V6": (13.0, 0.0, -4.0),
}
ELECTRODE_ORDER = ("LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6")


def heart_frame(apex_dir_body=(0.55, 0.45, -0.70), septal_hint_body=(-1.0, 0.6, 0.0)):
    """Mesh axes in body coordinates: mesh -x = apex direction, mesh +y = septal."""
    a = np.asarray(apex_dir_body, float)
    a /= np.linalg.norm(a)
    s = np.asarray(septal_hint_body, float)
    s -= (s @ a) * a
    s /= np.linalg.norm(s)
    ex, ey = -a, s
    return np.vstack([ex, ey, np.cross(ex, ey)])


def electrode_positions(lv_centre_m, frame=None):
    """10 x 3 array (m), mesh coordinates, order LA, RA, LL, RL, V1..V6."""
    R = heart_frame() if frame is None else frame
    return np.array([np.asarray(lv_centre_m) + R @ (np.asarray(BODY_ELECTRODES_CM[k]) * 1e-2)
                     for k in ELECTRODE_ORDER])


# ---------------------------------------------------------------------------- case assembly
def ep_node_fields(geo, markers, activation: Activation, celltypes: CellTypes):
    """Transmural coordinate, cell types and LAT on P1."""
    import dolfinx
    mesh = geo.mesh
    t = transmural_coordinate(mesh, geo.ffun, markers["LV"][0], markers["EPI"][0])
    W = t.function_space
    x_dof = W.tabulate_dof_coordinates()
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    tets = np.asarray(W.dofmap.list[:ncells], dtype=np.int64)
    f0_cells = geo.f0.x.array.reshape(-1, 3)[geo.f0.function_space.dofmap.list[:ncells, 0]]
    mesh.topology.create_connectivity(tdim - 1, tdim)
    endo_dofs = np.unique(dolfinx.fem.locate_dofs_topological(W, tdim - 1, geo.ffun.find(markers["LV"][0])))
    lat, roots = activation_times(x_dof, tets, f0_cells, endo_dofs, activation)
    ct = celltypes_from_transmural(t.x.array, celltypes)
    fields = {}
    for name, vals in (("lat_ms", lat), ("celltype", ct), ("sf_IKs", np.ones_like(lat)),
                       ("transmural", t.x.array.copy())):
        u = dolfinx.fem.Function(W, name=name)
        u.x.array[:] = vals
        fields[name] = u
    return fields, x_dof[roots]


def cavity_volume_ml(mesh, ffun, tag):
    import dolfinx
    import ufl
    X, N = ufl.SpatialCoordinate(mesh), ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ffun)
    return dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(X, N) * ds(tag))) * 1e6


def geometry_report(geo, markers) -> dict:
    """Reference-configuration metrics for checking physiology."""
    import dolfinx
    import ufl
    mesh = geo.mesh
    x = mesh.geometry.x
    tdim = mesh.topology.dim
    vol_myo = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(domain=mesh))) * 1e6
    V_cav = cavity_volume_ml(mesh, geo.ffun, markers["LV"][0])
    endo = np.unique(dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, geo.ffun.find(markers["LV"][0])))
    epi = np.unique(dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, geo.ffun.find(markers["EPI"][0])))
    xe, xp = x[endo], x[epi]
    L_endo = -xe[:, 0].min()
    sl = lambda pts, x0, tol: pts[np.abs(pts[:, 0] - x0) < tol]
    r_endo_mid = np.median(np.linalg.norm(sl(xe, -0.5 * L_endo, 0.08 * L_endo)[:, 1:], axis=1))
    r_epi_mid = np.median(np.linalg.norm(sl(xp, -0.5 * L_endo, 0.08 * L_endo)[:, 1:], axis=1))
    r_endo_base = np.max(np.linalg.norm(sl(xe, 0.0, 0.03 * L_endo)[:, 1:], axis=1))
    return {
        "cavity_volume_ml": V_cav,
        "wall_volume_ml": vol_myo,
        "lv_mass_g": vol_myo * 1.05,
        "cavity_long_axis_mm": L_endo * 1e3,
        "epi_long_axis_mm": -xp[:, 0].min() * 1e3,
        "mid_cavity_diameter_mm": 2 * r_endo_mid * 1e3,
        "basal_cavity_diameter_mm": 2 * r_endo_base * 1e3,
        "mid_wall_thickness_mm": (r_epi_mid - r_endo_mid) * 1e3,
        "apical_wall_thickness_mm": (xe[:, 0].min() - xp[:, 0].min()) * 1e3,
        "sphericity_index": (2 * r_endo_mid) / L_endo,
        "mass_to_volume_g_per_ml": vol_myo * 1.05 / V_cav,
        "cells": int(mesh.topology.index_map(tdim).size_local),
        "vertices": int(x.shape[0]),
    }


def helix_angles(geo, transmural):
    """Median helix angle (deg) in endo/mid/epi bands of the mid-ventricle. Sign is the
    geometric handedness: > 0 right-handed, < 0 left-handed. Healthy LV: right-handed
    subendocardium, left-handed subepicardium (Streeter)."""
    import dolfinx
    mesh = geo.mesh
    nc = mesh.topology.index_map(3).size_local
    mid = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(nc, dtype=np.int32))
    f = geo.f0.x.array.reshape(-1, 3)[geo.f0.function_space.dofmap.list[:nc, 0]]
    tc = transmural.x.array[transmural.function_space.dofmap.list[:nc]].mean(axis=1)
    r = mid[:, 1:] / np.linalg.norm(mid[:, 1:], axis=1, keepdims=True)
    e_theta = np.column_stack([np.zeros(nc), -r[:, 1], r[:, 0]])  # x_hat cross r_hat
    L = -mid[:, 0].min()
    sel = (mid[:, 0] < -0.2 * L) & (mid[:, 0] > -0.7 * L)
    ha = (np.degrees(np.arctan2(f[:, 0], np.sum(f * e_theta, axis=1))) + 90.0) % 180.0 - 90.0
    return {lab: float(np.median(ha[sel & (tc >= lo) & (tc <= hi)]))
            for lo, hi, lab in ((0.0, 0.25, "endo"), (0.4, 0.6, "mid"), (0.75, 1.0, "epi"))}


def write_case(outdir: Path, geo, markers, fields, electrodes_m, info: dict):
    """Save the dolfinx geometry folder, EP node fields, electrodes and a VTK preview."""
    import io4dolfinx
    import cardiac_geometries.geometry
    import dolfinx
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mesh = geo.mesh
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    cfun = dolfinx.mesh.meshtags(mesh, tdim, np.arange(ncells, dtype=np.int32), np.ones(ncells, dtype=np.int32))
    cfun.name = "Cell tags"
    cardiac_geometries.geometry.Geometry(
        mesh=mesh, markers=dict(markers), ffun=geo.ffun, cfun=cfun,
        f0=geo.f0, s0=geo.s0, n0=geo.n0, info=info).save_folder(outdir)
    for name, u in fields.items():
        io4dolfinx.write_function(filename=outdir / "ep_node_fields.bp", u=u, name=name)
    np.savetxt(outdir / "electrodes_m.csv", electrodes_m, delimiter=",",
               header="rows: " + ", ".join(ELECTRODE_ORDER) + " (m, mesh coordinates)")
    _write_vtk(outdir / "preview.vtk", mesh, fields, geo.f0)


def _write_vtk(path, mesh, fields, f0):
    ncells = mesh.topology.index_map(3).size_local
    W = next(iter(fields.values())).function_space
    x = W.tabulate_dof_coordinates()
    tets = np.asarray(W.dofmap.list[:ncells])
    fc = f0.x.array.reshape(-1, 3)[f0.function_space.dofmap.list[:ncells, 0]]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nLV ellipsoid\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(x)} double\n")
        np.savetxt(f, x)
        f.write(f"CELLS {ncells} {5 * ncells}\n")
        np.savetxt(f, np.hstack([np.full((ncells, 1), 4), tets]), fmt="%d")
        f.write(f"CELL_TYPES {ncells}\n")
        np.savetxt(f, np.full(ncells, 10), fmt="%d")
        f.write(f"CELL_DATA {ncells}\nVECTORS f0 double\n")
        np.savetxt(f, fc)
        f.write(f"POINT_DATA {len(x)}\n")
        for name, u in fields.items():
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, u.x.array)