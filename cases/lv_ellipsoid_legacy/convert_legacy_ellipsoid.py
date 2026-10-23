# convert_legacy_ellipsoid.py
# Legacy simcardems (dolfin, cardiac-geometries 0.3.9) LV ellipsoid -> physcardems case.
# Metres, basal plane at x = 0, DG0 fibres (cell mean of Quadrature_3, re-orthonormalised),
# EP fields and electrodes from physcardems.ellipsoid. This is the Land 2015 benchmark
# ellipsoid (~2.2 mL cavity), kept for comparison with legacy runs, not for physiology.
# Usage: python3 cases/lv_ellipsoid_legacy/convert_legacy_ellipsoid.py lv_ellipsoid.h5
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import basix.ufl
import dolfinx
from mpi4py import MPI

from physcardems import ellipsoid as ell

assert MPI.COMM_WORLD.size == 1, "run this serially"
comm = MPI.COMM_WORLD
ap = argparse.ArgumentParser()
ap.add_argument("h5")
ap.add_argument("--outdir", default=Path(__file__).resolve().parent)
args = ap.parse_args()
casedir = Path(args.outdir)
geodir = casedir / "geometry"
if geodir.exists():
    shutil.rmtree(geodir)

with h5py.File(args.h5, "r") as f:
    x = f["mesh/coordinates"][:].astype(float)
    topo = f["mesh/topology"][:].astype(np.int64)
    cell_indices = f["mesh/cell_indices"][:]
    ffun_topo = f["meshfunctions/ffun/topology"][:].astype(np.int64)
    ffun_vals = f["meshfunctions/ffun/values"][:].astype(np.int32)
    markers_raw = {k: tuple(int(i) for i in f["markers"][k][:]) for k in f["markers"]}
    legacy_info = {k: (v[()].decode() if isinstance(v[()], bytes) else v[()].item()) for k, v in f["info"].items()}
    fib = {}
    for name in ("f0", "s0", "n0"):
        g = f["microstructure"][name]
        vec, cd = g["vector_0"][:], g["cell_dofs"][:].astype(np.int64)
        x_cd, cells = g["x_cell_dofs"][:].astype(np.int64), g["cells"][:].astype(np.int64)
        npc = int(x_cd[1] - x_cd[0])
        assert np.all(np.diff(x_cd) == npc) and npc % 3 == 0
        local = vec[cd].reshape(len(cells), 3, npc // 3)   # legacy VectorElement: blocked by component
        out = np.empty((len(cells), 3))
        out[cells] = local.mean(axis=2)
        fib[name] = out
print("legacy info:", legacy_info)

base_x = x[ffun_topo[ffun_vals == markers_raw["BASE"][0]]].reshape(-1, 3)[:, 0]
assert np.ptp(base_x) < 1e-9, "basal facets are not planar in x"
x[:, 0] -= base_x[0]
x *= 1e-3

element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = dolfinx.mesh.create_mesh(comm, topo, element, x)
tdim = mesh.topology.dim
markers = {"LV": markers_raw["ENDO"], "EPI": markers_raw["EPI"], "BASE": markers_raw["BASE"]}
keep = np.isin(ffun_vals, [v[0] for v in markers.values()])
ents, vals = dolfinx.io.distribute_entity_data(mesh, tdim - 1, ffun_topo[keep], ffun_vals[keep])
mesh.topology.create_connectivity(tdim - 1, tdim)
ffun = dolfinx.mesh.meshtags_from_entities(mesh, tdim - 1, dolfinx.graph.adjacencylist(ents), vals.astype(np.int32))
ffun.name = "Facet tags"

ncells = mesh.topology.index_map(tdim).size_local
orig = np.asarray(mesh.topology.original_cell_index)[:ncells]
row = {k: v[cell_indices] for k, v in fib.items()}
f0, s0, n0 = ell.orthonormalise(row["f0"][orig], row["s0"][orig])
n0[np.sum(n0 * row["n0"][orig], axis=1) < 0] *= -1
V = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
dofs = V.dofmap.list[:ncells, 0]
funcs = {}
for name, arr in (("f0", f0), ("s0", s0), ("n0", n0)):
    u = dolfinx.fem.Function(V, name=name)
    u.x.array.reshape(-1, 3)[dofs] = arr
    funcs[name] = u


class _Geo:
    pass


geo = _Geo()
geo.mesh, geo.ffun = mesh, ffun
geo.f0, geo.s0, geo.n0 = funcs["f0"], funcs["s0"], funcs["n0"]

fields, roots = ell.ep_node_fields(geo, markers, ell.Activation(), ell.CellTypes())
report = ell.geometry_report(geo, markers)
lv_centre = np.array([-0.5 * report["cavity_long_axis_mm"] * 1e-3, 0.0, 0.0])
info = {"source": Path(args.h5).name, "legacy_info": legacy_info, "units": "m",
        "fibre_space": "DG_0 (cell mean of legacy Quadrature_3, re-orthonormalised)",
        "note": "Land et al. 2015 benchmark ellipsoid; not human-sized"}
ell.write_case(geodir, geo, markers, fields, ell.electrode_positions(lv_centre), info)
report["helix_angle_deg_right_handed_positive"] = ell.helix_angles(geo, fields["transmural"])
report["lat_total_activation_ms"] = float(fields["lat_ms"].x.array.max())
(casedir / "geometry_report.json").write_text(json.dumps(report, indent=2))
(casedir / "case.toml").write_text(f'''# lv_ellipsoid_legacy: converted by convert_legacy_ellipsoid.py
name = "lv_ellipsoid_legacy"

[geometry]
dir = "geometry"
myocardium_cell_tags = [1]
cavities = ["LV"]
base_normal_axis = 0

[geometry.tags]
LV = {markers["LV"][0]}
EPI = {markers["EPI"][0]}
BASE = {markers["BASE"][0]}

[ecg]
electrodes = "geometry/electrodes_m.csv"
unit_scale = 1.0
''')
print(json.dumps(report, indent=2))
print(f"max |f.s| {np.abs(np.sum(f0 * s0, axis=1)).max():.1e}; "
      f"exterior facets {len(dolfinx.mesh.exterior_facet_indices(mesh.topology))}, tagged {len(ffun.indices)}")
