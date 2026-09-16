from mpi4py import MPI
from pathlib import Path
import h5py
import numpy as np
import basix.ufl
import dolfinx
import ufl
from scipy.spatial import cKDTree
import cardiac_geometries.geometry

assert MPI.COMM_WORLD.size == 1, "run this serially"
comm = MPI.COMM_WORLD
fname = "rodero_05_coarse_4mm.h5"
outdir = Path("rodero_05_dolfinx")
MM_TO_M = 1e-3


# ---------------------------------------------------------------- helpers
def write_facets_vtk(path, mesh, ffun):
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    x = mesh.geometry.x
    tri = dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, ffun.indices)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nfacet tags\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(x)} double\n")
        np.savetxt(f, x)
        f.write(f"CELLS {len(tri)} {4 * len(tri)}\n")
        np.savetxt(f, np.hstack([np.full((len(tri), 1), 3), tri]), fmt="%d")
        f.write(f"CELL_TYPES {len(tri)}\n")
        np.savetxt(f, np.full(len(tri), 5), fmt="%d")
        f.write(f"CELL_DATA {len(tri)}\nSCALARS tag int 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, ffun.values, fmt="%d")


def cell_vectors(mesh, u):
    ncells = mesh.topology.index_map(mesh.topology.dim).size_local
    dofs = u.function_space.dofmap.list[:ncells, 0]
    return u.x.array.reshape(-1, 3)[dofs]


def centroids(mesh):
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    return dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(ncells, dtype=np.int32))


def write_fibres_vtk(path, mesh, funcs):
    mid = centroids(mesh)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nfibres\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(mid)} double\n")
        np.savetxt(f, mid)
        f.write(f"POINT_DATA {len(mid)}\n")
        for name, u in funcs.items():
            f.write(f"VECTORS {name} double\n")
            np.savetxt(f, cell_vectors(mesh, u))


# ---------------------------------------------------------------- read legacy h5
with h5py.File(fname, "r") as f:
    x = f["mesh/coordinates"][:] * MM_TO_M
    topo = f["mesh/topology"][:].astype(np.int64)
    cell_indices = f["mesh/cell_indices"][:]
    ffun_x = f["meshfunctions/ffun/coordinates"][:] * MM_TO_M
    ffun_topo = f["meshfunctions/ffun/topology"][:].astype(np.int64)
    ffun_vals = f["meshfunctions/ffun/values"][:]
    markers_raw = {k: f["markers"][k][:].tolist() for k in f["markers"]}
    fib = {}
    for name in ("f0", "s0", "n0"):
        g = f["microstructure"][name]
        vec = g["vector_0"][:]
        cells = g["cells"][:].astype(np.int64)
        x_cd = g["x_cell_dofs"][:].astype(np.int64)
        cd = g["cell_dofs"][:].astype(np.int64)
        assert np.all(np.diff(x_cd) == 3), f"{name}: expected 3 dofs per cell"
        out = np.empty((len(cells), 3))
        out[cells] = vec[cd].reshape(-1, 3)
        fib[name] = out

print("markers in file:", markers_raw)
print("ffun coordinates identical to mesh coordinates:", np.allclose(ffun_x, x))
print("cell_indices is identity:", np.array_equal(cell_indices, np.arange(len(cell_indices))))
row_fib = {k: v[cell_indices] for k, v in fib.items()}

# ---------------------------------------------------------------- build dolfinx objects
element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = dolfinx.mesh.create_mesh(comm, topo, element, x)
tdim = mesh.topology.dim

rename = {"ENDO_LV": "LV", "ENDO_RV": "RV", "EPI": "EPI", "BASE": "BASE"}
markers = {rename[k]: v for k, v in markers_raw.items()}
valid_ids = [v[0] for v in markers.values()]
keep = np.isin(ffun_vals.astype(np.int64), valid_ids)
ents, vals = dolfinx.io.distribute_entity_data(
    mesh, tdim - 1, ffun_topo[keep], ffun_vals[keep].astype(np.int32)
)
mesh.topology.create_connectivity(tdim - 1, tdim)
ffun = dolfinx.mesh.meshtags_from_entities(
    mesh, tdim - 1, dolfinx.graph.adjacencylist(ents), vals.astype(np.int32)
)
ffun.name = "Facet tags"

V = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
ncells = mesh.topology.index_map(tdim).size_local
orig = np.asarray(mesh.topology.original_cell_index)[:ncells]
dofs = V.dofmap.list[:ncells, 0]
funcs = {}
for name in ("f0", "s0", "n0"):
    u = dolfinx.fem.Function(V, name=name)
    u.x.array.reshape(-1, 3)[dofs] = row_fib[name][orig]
    funcs[name] = u

# ---------------------------------------------------------------- checks
print(f"\ncells={ncells}, vertices={mesh.geometry.x.shape[0]}")
print("bbox (m) min", mesh.geometry.x.min(axis=0), "max", mesh.geometry.x.max(axis=0))
ext = dolfinx.mesh.exterior_facet_indices(mesh.topology)
print(f"exterior facets={len(ext)}, tagged facets={len(ffun.indices)}, "
      f"untagged exterior={len(np.setdiff1d(ext, ffun.indices))}")
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ffun)
for k, (mid, _) in markers.items():
    n = int(np.sum(ffun.values == mid))
    area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ds(mid)))
    print(f"  {k:5s} id={mid} facets={n} area={area * 1e4:.1f} cm^2")
Fv = {k: cell_vectors(mesh, u) for k, u in funcs.items()}
for k, a in Fv.items():
    nrm = np.linalg.norm(a, axis=1)
    print(f"  |{k}| min={nrm.min():.6f} max={nrm.max():.6f}")
print(f"  max|f0.s0|={np.abs(np.sum(Fv['f0'] * Fv['s0'], axis=1)).max():.2e} "
      f"max|f0.n0|={np.abs(np.sum(Fv['f0'] * Fv['n0'], axis=1)).max():.2e}")
X = ufl.SpatialCoordinate(mesh)
N = ufl.FacetNormal(mesh)
for k in ("LV", "RV"):
    v = dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1 / 3 * ufl.dot(X, N) * ds(markers[k][0])))
    print(f"  {k} cavity volume = {v * 1e6:.1f} mL")

# ---------------------------------------------------------------- save
geo = cardiac_geometries.geometry.Geometry(
    mesh=mesh, markers=markers, ffun=ffun,
    f0=funcs["f0"], s0=funcs["s0"], n0=funcs["n0"],
    info={"source": fname, "units": "m", "fibre_space": "DG_0"},
)
geo.save_folder(outdir)
write_facets_vtk(outdir / "facets.vtk", mesh, ffun)
write_fibres_vtk(outdir / "fibres.vtk", mesh, funcs)
print(f"\nsaved to {outdir} (checkpoint + facets.vtk + fibres.vtk)")

# ---------------------------------------------------------------- round-trip check
geo2 = cardiac_geometries.geometry.Geometry.from_folder(comm, outdir)
ok = geo2.markers == markers
for k, (mid, _) in markers.items():
    n1 = int(np.sum(ffun.values == mid))
    n2 = int(np.sum(geo2.ffun.values == mid))
    ok &= n1 == n2
    if n1 != n2:
        print(f"  round-trip MISMATCH {k}: {n1} vs {n2} facets")
dist, idx = cKDTree(centroids(mesh)).query(centroids(geo2.mesh))
print(f"  round-trip centroid match: max distance {dist.max():.2e} m")
for name in ("f0", "s0", "n0"):
    diff = np.abs(cell_vectors(geo2.mesh, getattr(geo2, name)) - Fv[name][idx]).max()
    ok &= diff < 1e-12
    print(f"  round-trip {name} max abs difference {diff:.2e}")
print("ROUND TRIP OK" if ok else "ROUND TRIP FAILED")
