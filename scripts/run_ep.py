"""run_rodero_ep.py

Monodomain EP on rodero_05 (same coarse mesh as mechanics), no mechanics.
ToR-ORd-Land (full model, isometric: lmbda = 1, dLambda = 0) at every P1 node,
initialised from the steady-state cache for its cell type and stimulated at its
Eikonal activation time. Run serially.
"""
import argparse
import importlib.util
import json
import time as timer
from pathlib import Path
from scipy.spatial import cKDTree
from physcardems import ecg as pseudo_ecg

import numpy as np
from mpi4py import MPI
import dolfinx
import io4dolfinx
import beat
import cardiac_geometries.geometry

comm = MPI.COMM_WORLD
assert comm.size == 1, "run this serially"

ap = argparse.ArgumentParser()
ap.add_argument("--geodir", default="./rodero_05_dolfinx_v2")
ap.add_argument("--ssdir", default="./steady_state_pcl800")
ap.add_argument("--outdir", default="rodero-ep")
ap.add_argument("--t-end", type=float, default=800.0)       # ms
ap.add_argument("--dt", type=float, default=0.05)           # ms
ap.add_argument("--save-every", type=float, default=2.0)    # ms
ap.add_argument("--pcl", type=float, default=800.0)         # ms
ap.add_argument("--conductivities", default="Niederer")     # beat presets: Niederer, Bishop, Potse
ap.add_argument("--electrodes", default="./rodero_05_fine_nodefield_electrode_xyz.csv")
ap.add_argument("--electrode-unit-scale", type=float, default=1e-2)  # file in cm, mesh in m
ap.add_argument("--dt-ecg", type=float, default=1.0)                 # ms
ap.add_argument("--ecg-leads", choices=["standard", "legacy"], default="standard")
ap.add_argument("--ecg-conductivity", action="store_true")           # weight grad(v) by M (Gima-Rudy)
ap.add_argument("--lat-shift", default="min")  # ms subtracted from LAT; "min" starts activation at t = 0
args = ap.parse_args()

GEODIR, SSDIR, OUTDIR = Path(args.geodir), Path(args.ssdir), Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)
CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}

# ------------------------------------------------------------------ geometry and node fields
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=GEODIR)
mesh = geo.mesh
W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
fields = {}
for name in ("lat_ms", "celltype", "sf_IKs"):
    fields[name] = dolfinx.fem.Function(W, name=name)
    io4dolfinx.read_function(filename=GEODIR / "ep_node_fields.bp", u=fields[name], name=name)
# lat = fields["lat_ms"].x.array.copy()
lat_raw = fields["lat_ms"].x.array.copy()
lat_shift = float(lat_raw.min()) if args.lat_shift == "min" else float(args.lat_shift)
lat = lat_raw - lat_shift
print(f"LAT shifted by -{lat_shift:.1f} ms")
ct = np.rint(fields["celltype"].x.array).astype(int)
N = lat.size
print(f"{N} ODE nodes | LAT {lat.min():.1f} to {lat.max():.1f} ms | "
      + ", ".join(f"{CELLTYPES[c]} {int(np.sum(ct == c))}" for c in CELLTYPES))

# ------------------------------------------------------------------ cell model, initial states, parameters
spec = importlib.util.spec_from_file_location("torord_land_full", SSDIR / "torord_land_full.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
state_names = sorted(model.state, key=model.state.get)
iv, ica = model.state_index("v"), model.state_index("cai")
iTa = model.monitor_index("Ta")

y0 = np.zeros((len(state_names), N))
for c, cname in CELLTYPES.items():
    ss = json.loads((SSDIR / f"celltype{c}_{cname}.json").read_text())["states"]
    y0[:, ct == c] = np.array([ss[s] for s in state_names])[:, None]

P = np.repeat(model.init_parameter_values(i_Stim_Period=args.pcl, lmbda=1.0, dLambda=0.0)[:, None], N, axis=1)
P[model.parameter_index("celltype")] = ct.astype(float)
P[model.parameter_index("i_Stim_Start")] = lat

# ------------------------------------------------------------------ monodomain
time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
cond = beat.conductivities.default_conductivities(args.conductivities)
M = beat.conductivities.define_conductivity_tensor(f0=geo.f0, **cond)
C_m = (1.0 * beat.units.ureg("uF/cm**2")).to("uF/m**2").magnitude  # mesh is in metres
pde = beat.MonodomainModel(time=time, mesh=mesh, M=M, I_s=None, C_m=C_m)

v_ode = dolfinx.fem.Function(W, name="v_ode")
ode = beat.odesolver.DolfinODESolver(
    v_ode=v_ode,
    v_pde=pde.state,
    fun=model.generalized_rush_larsen,
    init_states=y0,
    parameters=P,
    num_states=y0.shape[0],
    v_index=iv,
)
solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

# ------------------------------------------------------------------ pseudo-ECG
electrodes = pseudo_ecg.load_electrodes(args.electrodes, args.electrode_unit_scale)
dist, _ = cKDTree(mesh.geometry.x).query(np.array(list(electrodes.values())))
print("electrode distance to nearest mesh node (mm): "
      + ", ".join(f"{k} {d * 1e3:.0f}" for k, d in zip(electrodes, dist)))
build = pseudo_ecg.standard_leads if args.ecg_leads == "standard" else pseudo_ecg.legacy_leads
ecg_points, ecg_leads = build(electrodes)
ecg = pseudo_ecg.PseudoECG(pde.state, ecg_points, ecg_leads, sigma_b=1.0,
                           G=M if args.ecg_conductivity else None)
ecg_every = max(1, int(round(args.dt_ecg / args.dt)))
ecg_csv_every = int(round(50.0 / args.dt))

# ------------------------------------------------------------------ output
cai_fn = dolfinx.fem.Function(W, name="cai_mM")
ta_fn = dolfinx.fem.Function(W, name="Ta_kPa")
pde.state.name = "v"
vtx = dolfinx.io.VTXWriter(comm, OUTDIR / "ep.bp", [pde.state, cai_fn, ta_fn], engine="BP4")


def save(t):
    vals = ode.values
    cai_fn.x.array[:] = vals[ica]
    ta_fn.x.array[:] = model.monitor_values(t, vals, P)[iTa]
    vtx.write(t)


# ------------------------------------------------------------------ time loop
n_steps = int(round(args.t_end / args.dt))
save_every = max(1, int(round(args.save_every / args.dt)))
report_every = int(round(10.0 / args.dt))
act = np.full(N, np.nan)   # simulated activation time (v up through -40 mV)
rep = np.full(N, np.nan)   # simulated repolarisation time (v down through -70 mV)

save(0.0)
wall0 = timer.time()
for k in range(1, n_steps + 1):
    t0, t = (k - 1) * args.dt, k * args.dt
    solver.step((t0, t))
    v = ode.values[iv]
    new = np.isnan(act) & (v > -40.0)
    act[new] = t
    done = ~np.isnan(act) & np.isnan(rep) & (t > act + 20.0) & (v < -70.0)
    rep[done] = t
    if k % save_every == 0:
        save(t)
    if k % ecg_every == 0:
        ecg.compute(t)
    if k % ecg_csv_every == 0:
        ecg.save_csv(OUTDIR / "pseudo_ecg.csv")
        ecg.plot(OUTDIR / "pseudo_ecg.png")
    if k % report_every == 0:
        el = timer.time() - wall0
        print(f"t={t:6.1f} ms | v [{v.min():6.1f}, {v.max():6.1f}] mV | activated {np.mean(~np.isnan(act)):5.1%} "
              f"repolarised {np.mean(~np.isnan(rep)):5.1%} | {el:6.0f} s wall, {el / k * 1e3:.1f} ms/step")
vtx.close()
ecg.save_csv(OUTDIR / "pseudo_ecg.csv")
ecg.plot(OUTDIR / "pseudo_ecg.png")

# ------------------------------------------------------------------ activation / repolarisation summary
ok = ~np.isnan(act)
d = act[ok] - lat[ok]
print(f"\nactivated {ok.sum()}/{N} nodes; simulated minus Eikonal LAT: "
      f"median {np.median(d):.2f} ms, 5% {np.percentile(d, 5):.2f}, 95% {np.percentile(d, 95):.2f}, "
      f"min {d.min():.2f}, max {d.max():.2f}")
early = np.sum(d < -1.0)
print(f"nodes activating more than 1 ms before their own LAT (electrotonic): {early}")
apd = rep - act
for c, cname in CELLTYPES.items():
    m = (ct == c) & ~np.isnan(apd)
    if m.any():
        print(f"  {cname:4s}: activation-recovery interval median {np.median(apd[m]):.0f} ms "
              f"(5% {np.percentile(apd[m], 5):.0f}, 95% {np.percentile(apd[m], 95):.0f})")
print(f"repolarised {np.sum(~np.isnan(rep))}/{N} nodes by t={args.t_end:.0f} ms")

maps = {"lat_input_ms": lat, "lat_sim_ms": act, "rt_sim_ms": rep, "ari_sim_ms": apd, "celltype": ct.astype(float)}
np.savez(OUTDIR / "maps.npz", **maps)
map_funcs = []
for name, arr in maps.items():
    f = dolfinx.fem.Function(W, name=name)
    f.x.array[:] = np.nan_to_num(arr, nan=-1.0)
    map_funcs.append(f)
with dolfinx.io.VTXWriter(comm, OUTDIR / "maps.bp", map_funcs, engine="BP4") as w:
    w.write(0.0)
print(f"output in {OUTDIR}")