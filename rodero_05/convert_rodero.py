from mpi4py import MPI
from pathlib import Path
import shutil
import h5py
import numpy as np
import pandas as pd
import basix.ufl
import dolfinx
import ufl
import io4dolfinx
from scipy.spatial import cKDTree
import cardiac_geometries.geometry

assert MPI.COMM_WORLD.size == 1, "run this serially"
comm = MPI.COMM_WORLD

fname = "rodero_05_coarse_4mm.h5"
tv_file = "rodero_05_coarse_4mm_tv.npy"
lat_file = "rodero_05_coarse_4mm_lat.csv"
fine_xyz_file = "rodero_05_fine_xyz.csv"
fine_ct_file = "rodero_05_fine_nodefield_cell-type.csv"
fine_iks_file = "rodero_05_fine_nodefield_sf_IKs.csv"
outdir = Path("rodero_05_dolfinx_v2")
ep_fields_path = outdir / "ep_node_fields.bp"

MM_TO_M = 1e-3
CM_TO_M = 1e-2
ALYA_TO_TORORD_CELLTYPE = {1: 0, 2: 2, 3: 1}  # Alya 1=endo 2=mid 3=epi -> ToR-ORd 0=endo 1=epi 2=mid
TISSUE_NAMES = {1: "LV myo", 2: "RV myo", 7: "LV valve", 8: "RV valve", 9: "LV valve", 10: "RV valve"}

if outdir.exists():
    shutil.rmtree(outdir)


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


def write_volume_vtk(path, mesh, tissue, point_fields):
    tdim = mesh.topology.dim
    x = mesh.geometry.x
    ncells = mesh.topology.index_map(tdim).size_local
    tets = dolfinx.mesh.entities_to_geometry(mesh, tdim, np.arange(ncells, dtype=np.int32))
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ntissue and EP node fields\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(x)} double\n")
        np.savetxt(f, x)
        f.write(f"CELLS {ncells} {5 * ncells}\n")
        np.savetxt(f, np.hstack([np.full((ncells, 1), 4), tets]), fmt="%d")
        f.write(f"CELL_TYPES {ncells}\n")
        np.savetxt(f, np.full(ncells, 10), fmt="%d")
        f.write(f"CELL_DATA {ncells}\nSCALARS tissue int 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, tissue, fmt="%d")
        f.write(f"POINT_DATA {len(x)}\n")
        for name, vals in point_fields.items():
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, vals)


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


def tissue_array(mesh, cfun):
    ncells = mesh.topology.index_map(mesh.topology.dim).size_local
    arr = np.zeros(ncells, dtype=np.int32)
    arr[cfun.indices] = cfun.values
    return arr


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

# ---------------------------------------------------------------- read tissue types and EP node data
tv = np.load(tv_file).astype(np.int32)
assert len(tv) == len(topo), f"tissue types {len(tv)} != cells {len(topo)}"
row_tv = tv[cell_indices]
print("tissue types in file:", dict(zip(*np.unique(tv, return_counts=True))))

lat_df = pd.read_csv(lat_file).sort_values("node_id")
assert np.array_equal(lat_df["node_id"].to_numpy(), np.arange(len(x))), "LAT node ids do not match h5 nodes"
lat_h5 = lat_df["activation_time_ms"].to_numpy()

fine_x = pd.read_csv(fine_xyz_file, header=None).to_numpy() * CM_TO_M
fine_ct = np.loadtxt(fine_ct_file).astype(int)
fine_iks = np.loadtxt(fine_iks_file)
assert len(fine_x) == len(fine_ct) == len(fine_iks), "fine node files have different lengths"
unknown = set(np.unique(fine_ct)) - set(ALYA_TO_TORORD_CELLTYPE)
assert not unknown, f"unexpected Alya cell-type codes {unknown}"

d_fine, nearest_fine = cKDTree(fine_x).query(x)
ct_h5 = np.array([ALYA_TO_TORORD_CELLTYPE[c] for c in fine_ct[nearest_fine]], dtype=float)
iks_h5 = fine_iks[nearest_fine]
node_fields_h5 = {"lat_ms": lat_h5, "celltype": ct_h5, "sf_IKs": iks_h5}
tree_h5 = cKDTree(x)


def h5_node_index(points):
    d, idx = tree_h5.query(points)
    assert d.max() < 1e-9, f"exact node match failed, max distance {d.max():.3e} m"
    return idx


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

ncells = mesh.topology.index_map(tdim).size_local
orig = np.asarray(mesh.topology.original_cell_index)[:ncells]

V = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
dofs = V.dofmap.list[:ncells, 0]
funcs = {}
for name in ("f0", "s0", "n0"):
    u = dolfinx.fem.Function(V, name=name)
    u.x.array.reshape(-1, 3)[dofs] = row_fib[name][orig]
    funcs[name] = u

cfun = dolfinx.mesh.meshtags(mesh, tdim, np.arange(ncells, dtype=np.int32), row_tv[orig])
cfun.name = "Cell tags"

W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
dof_h5 = h5_node_index(W.tabulate_dof_coordinates())
node_funcs = {}
for name, vals_h5 in node_fields_h5.items():
    u = dolfinx.fem.Function(W, name=name)
    u.x.array[:] = vals_h5[dof_h5]
    node_funcs[name] = u
geom_h5 = h5_node_index(mesh.geometry.x)

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

tissue = tissue_array(mesh, cfun)
print("\ntissue types:")
for t, c in zip(*np.unique(tissue, return_counts=True)):
    print(f"  {t:2d} ({TISSUE_NAMES.get(int(t), '?')}): {c} cells")
f2c = mesh.topology.connectivity(tdim - 1, tdim)
print("tissue types of cells adjacent to each surface:")
for k, (mid, _) in markers.items():
    adj = np.array([f2c.links(fct)[0] for fct in ffun.find(mid)])
    print(f"  {k:5s}", dict(zip(*[a.tolist() for a in np.unique(tissue[adj], return_counts=True)])))

print("\nEP node fields:")
print(f"  LAT: min={lat_h5.min():.1f} ms, max={lat_h5.max():.1f} ms")
print(f"  nearest fine node distance (mm): median={np.median(d_fine) * 1e3:.2f}, "
      f"95%={np.percentile(d_fine, 95) * 1e3:.2f}, max={d_fine.max() * 1e3:.2f}")
ct_names = {0: "endo", 1: "epi", 2: "mid"}
print("  celltype counts (all nodes):",
      {ct_names[int(c)]: int(n) for c, n in zip(*np.unique(ct_h5, return_counts=True))})
for k in ("LV", "RV", "EPI"):
    surf_nodes = np.unique(dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, ffun.find(markers[k][0])))
    cts = ct_h5[geom_h5[surf_nodes]]
    print(f"  celltype on {k:3s} surface nodes:",
          {ct_names[int(c)]: int(n) for c, n in zip(*np.unique(cts, return_counts=True))})
for c in (0, 1, 2):
    print(f"  sf_IKs {ct_names[c]:4s}: 5%={np.percentile(iks_h5[ct_h5 == c], 5):.4f} "
          f"median={np.median(iks_h5[ct_h5 == c]):.4f} 95%={np.percentile(iks_h5[ct_h5 == c], 95):.4f}")

# ---------------------------------------------------------------- save
geo = cardiac_geometries.geometry.Geometry(
    mesh=mesh, markers=markers, ffun=ffun, cfun=cfun,
    f0=funcs["f0"], s0=funcs["s0"], n0=funcs["n0"],
    info={"source": fname, "units": "m", "fibre_space": "DG_0",
          "cell_tags": "tissue type (1 LV myo, 2 RV myo, 7-10 valve plugs)",
          "ep_node_fields": "ep_node_fields.bp: lat_ms, celltype (ToR-ORd 0 endo 1 epi 2 mid), sf_IKs"},
)
geo.save_folder(outdir)
for name, u in node_funcs.items():
    io4dolfinx.write_function(filename=ep_fields_path, u=u, name=name)

write_facets_vtk(outdir / "facets.vtk", mesh, ffun)
write_fibres_vtk(outdir / "fibres.vtk", mesh, funcs)
write_volume_vtk(outdir / "volume.vtk", mesh, tissue,
                 {name: vals_h5[geom_h5] for name, vals_h5 in node_fields_h5.items()})
print(f"\nsaved to {outdir} (checkpoint + ep_node_fields.bp + facets/fibres/volume .vtk)")

# ---------------------------------------------------------------- round-trip check
geo2 = cardiac_geometries.geometry.Geometry.from_folder(comm, outdir)
ok = geo2.markers == markers
for k, (mid, _) in markers.items():
    n1, n2 = int(np.sum(ffun.values == mid)), int(np.sum(geo2.ffun.values == mid))
    ok &= n1 == n2
    if n1 != n2:
        print(f"  round-trip MISMATCH {k}: {n1} vs {n2} facets")

dist, idx = cKDTree(centroids(mesh)).query(centroids(geo2.mesh))
print(f"  round-trip centroid match: max distance {dist.max():.2e} m")
for name in ("f0", "s0", "n0"):
    diff = np.abs(cell_vectors(geo2.mesh, getattr(geo2, name)) - Fv[name][idx]).max()
    ok &= diff < 1e-12
    print(f"  round-trip {name} max abs difference {diff:.2e}")

if geo2.cfun is None:
    print("  round-trip MISMATCH: cell tags not loaded")
    ok = False
else:
    tissue2 = tissue_array(geo2.mesh, geo2.cfun)
    n_bad = int(np.sum(tissue2 != tissue[idx]))
    ok &= n_bad == 0
    print(f"  round-trip tissue types: {n_bad} cells differ")

W2 = dolfinx.fem.functionspace(geo2.mesh, ("Lagrange", 1))
dof_h5_2 = h5_node_index(W2.tabulate_dof_coordinates())
for name, vals_h5 in node_fields_h5.items():
    u2 = dolfinx.fem.Function(W2, name=name)
    io4dolfinx.read_function(filename=ep_fields_path, u=u2, name=name)
    diff = np.abs(u2.x.array - vals_h5[dof_h5_2]).max()
    ok &= diff < 1e-12
    print(f"  round-trip {name} max abs difference {diff:.2e}")

print("ROUND TRIP OK" if ok else "ROUND TRIP FAILED")