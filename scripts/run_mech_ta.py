# run_mech_ta.py
# Mechanics driven by a prescribed active tension profile: no EP, no cell model.
# Ta(x, t) = analytical profile of (t - LAT(x)) per cell type (physcardems.prescribed),
# with the same geometry, passive material, valve plugs, boundary conditions and
# five-phase cycle controller as run_em.py. All settings come from a config file.
# Serial only.
#
# Usage: python3 scripts/run_mech_ta.py configs/ta_sweep/base.toml
from pathlib import Path
import argparse
import dataclasses
import json
import logging
import time as timer

import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import io4dolfinx
import pulse
import cardiac_geometries.geometry

from physcardems.cavity import ControlledCavityDynamicProblem, CavityMonitor
from physcardems.cycle import PHASE_NAMES, CycleParams, WindkesselParams, CavityState, BiVCycleController
from physcardems.prescribed_ta import PrescribedActive, celltype_profiles, CELLTYPES
from physcardems.plotting import LOG_HEADER, plot_summary, ejection_metrics
from physcardems.config import load_config, save_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mech_ta")
comm = MPI.COMM_WORLD
assert comm.size == 1, "serial only for now"

# ------------------------------------------------------------------ settings (from config)
ap = argparse.ArgumentParser(description="Mechanics with prescribed active tension")
ap.add_argument("config", help="simulation configuration (.toml or .json)")
args = ap.parse_args()
cfg, case = load_config(args.config)

GEODIR = case.path("geometry", "dir")
TAGS = case.get("geometry", "tags")
MYOCARDIUM_TAGS = tuple(case.get("geometry", "myocardium_cell_tags"))
OUTDIR = Path(cfg.get("outdir") or Path("runs") / case.name / Path(args.config).stem)

t = cfg["time"]
DT_MECH_MS, T_END_MS, PCL_MS = t["dt_mech_ms"], t["t_end_ms"], t["pcl_ms"]
OUTPUT_EVERY_MS, CHECKPOINT_EVERY_MS, PLOT_EVERY = t["output_every_ms"], t["checkpoint_every_ms"], t["plot_every"]
RESTART_FROM = cfg.get("restart", {}).get("from") or None

LAT_SHIFT_MS = cfg["activation"]["lat_shift_ms"]

m = cfg["mechanics"]
QUAD_DEGREE = m["quad_degree"]
KAPPA_PA = m["kappa_pa"]
VALVE_STIFFNESS_SCALE = m["valve_stiffness_scale"]
FIBRE_COMPRESSION_RESISTANCE = m["fibre_compression_resistance"]
REFERENCE_SCALE = m.get("reference_scale")
REFERENCE_LV_VOLUME_ML = m.get("reference_lv_volume_ml")
PRECONDITIONER_LAG = m["preconditioner_lag"]
MATERIAL_PARAMS = dict(m["material"])
for k, s in m.get("material_scales", {}).items():   # e.g. { a = 1.5 } for sweeps
    MATERIAL_PARAMS[k] *= s

ACTIVE = cfg["active"]
TA_D_KPA = ACTIVE.get("Ta_d_kpa", 0.0)
PROFILES = celltype_profiles(ACTIVE, PCL_MS)
FORMULATION = {"stretch": pulse.ActiveStressFormulation.stretch,
               "invariant": pulse.ActiveStressFormulation.invariant}[ACTIVE.get("formulation", "stretch")]

KPA_MS_PER_ML = 1e3 * 1e-3 / 1e-6
ML_PER_MS = 1e-6 / 1e-3
MMHG_S_PER_ML = 133.322 / 1e-6
ML_PER_MMHG = 1e-6 / 133.322


def cycle_params(c):
    wk = c["windkessel"]
    return CycleParams(
        t_zero=c["t_zero_s"], preload_pressure=c["preload_pressure_pa"],
        t_end_diastole=c["t_end_diastole_s"], p_end_diastole=c["p_end_diastole_pa"],
        p_fill=c["p_fill_pa"], period=PCL_MS / 1e3,
        windkessel=WindkesselParams(
            p_init=wk["p_init_pa"],
            resistance=wk["resistance_mmhg_s_per_ml"] * MMHG_S_PER_ML,
            compliance=wk["compliance_ml_per_mmhg"] * ML_PER_MMHG,
            characteristic_impedance=wk["characteristic_impedance_mmhg_s_per_ml"] * MMHG_S_PER_ML),
        filling_rate=c["filling_rate_ml_per_ms"] * ML_PER_MS,
    )


lv_params = cycle_params(cfg["circulation"]["lv"])
rv_params = cycle_params(cfg["circulation"]["rv"])

OUTDIR.mkdir(parents=True, exist_ok=True)
save_config(cfg, OUTDIR / "config_used.json", case=str(case.root), active_profiles=PROFILES)
CKPT_DIR = OUTDIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
for name, p in PROFILES.items():
    logger.info(f"Ta profile {name}: " + ", ".join(f"{k} {v:.4g}" for k, v in p.items()))


def ho_params(p):
    kpa = lambda v: pulse.Variable(v, "kPa")
    dim = lambda v: pulse.Variable(v, "dimensionless")
    return dict(a=kpa(p["a"]), b=dim(p["b"]), a_f=kpa(p["a_f"]), b_f=dim(p["b_f"]),
                a_s=kpa(p["a_s"]), b_s=dim(p["b_s"]), a_fs=kpa(p["a_fs"]), b_fs=dim(p["b_fs"]))


class ScaledModel:
    """Multiplies a pulse material or active model's stress and energy by a spatial field."""

    def __init__(self, model, scale):
        self._model = model
        self._scale = scale

    def __getattr__(self, name):
        return getattr(self._model, name)

    def S(self, C, *args, **kwargs):
        return self._scale * self._model.S(C, *args, **kwargs)

    def P(self, F, *args, **kwargs):
        return self._scale * self._model.P(F, *args, **kwargs)

    def strain_energy(self, *args, **kwargs):
        return self._scale * self._model.strain_energy(*args, **kwargs)


# ------------------------------------------------------------------ geometry, reference scaling, tissue masks
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=GEODIR)
for name, tag in TAGS.items():
    assert int(geo.markers[name][0]) == tag, f"{name}: expected tag {tag}, file has {geo.markers[name]}"
assert geo.cfun is not None, f"geometry in {GEODIR} has no cell tags"


def cavity_volume_ml(marker):
    X = ufl.SpatialCoordinate(geo.mesh)
    N = ufl.FacetNormal(geo.mesh)
    ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)
    form = dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(X, N) * ds(TAGS[marker]))
    return dolfinx.fem.assemble_scalar(form) * 1e6


assert (REFERENCE_SCALE is None) != (REFERENCE_LV_VOLUME_ML is None), \
    "set exactly one of reference_scale or reference_lv_volume_ml"
V_lv_imaged, V_rv_imaged = cavity_volume_ml("LV"), cavity_volume_ml("RV")
ref_scale = (REFERENCE_SCALE if REFERENCE_SCALE is not None
             else (REFERENCE_LV_VOLUME_ML / V_lv_imaged) ** (1.0 / 3.0))
xg = geo.mesh.geometry.x
centroid = xg.mean(axis=0)
xg[:] = centroid + ref_scale * (xg - centroid)
logger.info(f"Reference scaling {ref_scale:.4f}: LV {V_lv_imaged:.1f} -> {cavity_volume_ml('LV'):.1f} mL, "
            f"RV {V_rv_imaged:.1f} -> {cavity_volume_ml('RV'):.1f} mL")

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": QUAD_DEGREE})
mesh = geometry.mesh

DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
myo_mask = dolfinx.fem.Function(DG0, name="myocardium")
stiffness_scale = dolfinx.fem.Function(DG0, name="stiffness_scale")
tag_dofs = DG0.dofmap.list[geo.cfun.indices, 0]
is_myo = np.isin(geo.cfun.values, MYOCARDIUM_TAGS)
myo_mask.x.array[tag_dofs] = is_myo.astype(float)
stiffness_scale.x.array[tag_dofs] = np.where(is_myo, 1.0, VALVE_STIFFNESS_SCALE)
logger.info(f"Valve plug cells: {int(np.sum(~is_myo))} (Ta = 0, passive stiffness x{VALVE_STIFFNESS_SCALE})")

# ------------------------------------------------------------------ activation and cell type (node fields)
W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
fields = {}
for name in ("lat_ms", "celltype"):
    fields[name] = dolfinx.fem.Function(W, name=name)
    io4dolfinx.read_function(filename=GEODIR / "ep_node_fields.bp", u=fields[name], name=name)
lat = fields["lat_ms"].x.array.copy() - LAT_SHIFT_MS
ct = np.rint(fields["celltype"].x.array).astype(int)
logger.info(f"Activation: LAT {lat.min():.1f} to {lat.max():.1f} ms; nodes per cell type: "
            + ", ".join(f"{CELLTYPES[c]} {int(np.sum(ct == c))}" for c in CELLTYPES))

# ------------------------------------------------------------------ mechanics
active = PrescribedActive(f0=geo.f0, mesh=mesh, lat_ms=lat, celltype=ct, profiles=PROFILES,
                          ta_d_kpa=TA_D_KPA, pcl_ms=PCL_MS, formulation=FORMULATION)
material = ScaledModel(
    pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **ho_params(MATERIAL_PARAMS),
                         use_heaviside=not FIBRE_COMPRESSION_RESISTANCE,
                         use_subplus=not FIBRE_COMPRESSION_RESISTANCE),
    stiffness_scale,
)
model = pulse.CardiacModel(
    material=material,
    active=pulse.active_model.Passive(),  # active stress is added at the true displacement by the problem
    compressibility=pulse.compressibility.Compressible2(kappa=pulse.Variable(KAPPA_PA, "Pa")),
    viscoelasticity=pulse.viscoelasticity.Viscous(),
)
alpha_epi = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1e8)), "Pa / m")
beta_epi = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m")
bcs = pulse.BoundaryConditions(robin=(
    pulse.RobinBC(value=alpha_epi, marker=TAGS["EPI"]),
    pulse.RobinBC(value=beta_epi, marker=TAGS["EPI"], damping=True),
))
cavities = [
    pulse.problem.Cavity(marker="LV", volume=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))),
    pulse.problem.Cavity(marker="RV", volume=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))),
]
params = pulse.problem.DynamicProblem.default_parameters()
params["dt"] = pulse.Variable(DT_MECH_MS / 1e3, "s")
params["petsc_options"] = {
    **params["petsc_options"],
    "snes_monitor": None,
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "ksp_max_it": 100,
    "ksp_rtol": 1e-6,
    "ksp_atol": 1e-14,
    "ksp_converged_reason": None,
    "snes_lag_preconditioner": PRECONDITIONER_LAG,
    "snes_lag_preconditioner_persists": True,
}
logger.info("Building mechanics problem (form compilation can take several minutes)")
problem = ControlledCavityDynamicProblem(model=model, geometry=geometry, bcs=bcs, cavities=cavities,
                                         parameters=params, true_u_active=ScaledModel(active, myo_mask))
active.register(problem.u)
monitor = CavityMonitor(problem)
states = {"LV": CavityState(name="LV", params=lv_params), "RV": CavityState(name="RV", params=rv_params)}
controller = BiVCycleController(problem, monitor, states, restore_lag=PRECONDITIONER_LAG)
logger.info("Mechanics problem ready")

# ------------------------------------------------------------------ diagnostics and output
X_ref = np.array([[0.25, 0.25, 0.25], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dolfinx.default_real_type)
J_expr = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(problem.u)), X_ref)
local_cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)


def detF_stats():
    J = J_expr.eval(mesh, local_cells)
    return float(J.min()), float(J.max()), int(np.sum(np.min(J, axis=1) <= 0.0))


DG1 = active.function_space
myo_dg1 = dolfinx.fem.Function(DG1)
myo_dg1.interpolate(myo_mask)
myo_dofs = myo_dg1.x.array > 0.5
myo_nodes = np.zeros(W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts, dtype=bool)
w_test = ufl.TestFunction(W)
dx_out = ufl.Measure("dx", domain=mesh)
lumped_mass = dolfinx.fem.assemble_vector(dolfinx.fem.form(w_test * dx_out)).array.copy()
myo_frac = dolfinx.fem.assemble_vector(dolfinx.fem.form(myo_mask * w_test * dx_out)).array / lumped_mass
myo_nodes[:] = myo_frac > 0.5

Wv = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
u_out = dolfinx.fem.Function(Wv, name="u")
ta_out = dolfinx.fem.Function(W, name="Ta_kPa")
lam_out = dolfinx.fem.Function(W, name="lambda")
lam_form = dolfinx.fem.form(active.lmbda * w_test * dx_out)
ct_out = dolfinx.fem.Function(W, name="celltype")
ct_out.x.array[:] = ct
lat_out = dolfinx.fem.Function(W, name="lat_ms")
lat_out.x.array[:] = lat
myo_out = dolfinx.fem.Function(W, name="myocardium")
myo_out.x.array[:] = myo_frac
vtx_all = dolfinx.io.VTXWriter(comm, OUTDIR / "simulation.bp",
                               [u_out, ta_out, lam_out, ct_out, lat_out, myo_out], engine="BP4")


def write_outputs(t_ms):
    u_out.interpolate(problem.u)
    ta_out.x.array[:] = active.Ta.x.array * myo_frac / 1e3
    lam_out.x.array[:] = dolfinx.fem.assemble_vector(lam_form).array / lumped_mass
    vtx_all.write(t_ms)


def write_checkpoint(t_ms):
    base = CKPT_DIR / f"t_{t_ms:08.3f}"
    for i, (name, f) in enumerate((("u", problem.u), ("v", problem.v_old), ("a", problem.a_old))):
        mode = io4dolfinx.FileMode.write if i == 0 else io4dolfinx.FileMode.append
        io4dolfinx.write_function_on_input_mesh(f"{base}.bp", f, time=0.0, name=name, mode=mode, backend="adios2")
    data = {
        "t_ms": t_ms, "dt_mech_ms": DT_MECH_MS, "reference_scale": ref_scale,
        "pressures": {m: monitor.pressure(m) for m in states},
        "states": {m: {k: v for k, v in dataclasses.asdict(s).items() if k not in ("name", "params")}
                   for m, s in states.items()},
    }
    Path(f"{base}.json").write_text(json.dumps(data, indent=2))
    logger.info(f"Checkpoint written: {base}")


def save_log(arr):
    np.savetxt(OUTDIR / "log.csv", arr, delimiter=",", header=LOG_HEADER, comments="")
    plot_summary(arr, OUTDIR / "summary.png", title=f"{OUTDIR}  (prescribed Ta, reference scale {ref_scale:.3f})")


def save_metrics(arr):
    """Per-run scalar outputs, for the sweep."""
    out = {"t_end_reached_ms": float(arr[-1, 0]) if len(arr) else 0.0}
    for side, (iv, iph, ip) in {"lv": (5, 4, 6), "rv": (9, 8, 10)}.items():
        edv, esv, ef, complete = ejection_metrics(arr[:, iv], arr[:, iph])
        out.update({f"{side}_edv_ml": edv, f"{side}_esv_ml": esv, f"{side}_ef_pct": ef,
                    f"{side}_sv_ml": edv - esv, f"{side}_ejection_complete": complete,
                    f"{side}_p_max_kpa": float(np.max(arr[:, ip])) if len(arr) else np.nan})
    (OUTDIR / "metrics.json").write_text(json.dumps(out, indent=2, default=float))


# ------------------------------------------------------------------ start or restart
if RESTART_FROM is None:
    active.set_time(0.0)
    problem.solve()
    controller.initialize(0.0, {m: monitor.volume(m) for m in states})
    t_start_ms = 0.0
else:
    base = str(RESTART_FROM)
    data = json.loads(Path(f"{base}.json").read_text())
    assert np.isclose(data["dt_mech_ms"], DT_MECH_MS)
    assert np.isclose(data.get("reference_scale", 1.0), ref_scale), "checkpoint used a different reference scaling"
    for name, f in (("u", problem.u), ("v", problem.v_old), ("a", problem.a_old)):
        io4dolfinx.read_function(f"{base}.bp", f, time=0.0, name=name, backend="adios2")
        f.x.scatter_forward()
    problem.u_old.x.array[:] = problem.u.x.array
    for m, s in states.items():
        for k, v in data["states"][m].items():
            setattr(s, k, v)
        monitor.init_pressure(m, data["pressures"][m])
    t_start_ms = data["t_ms"]
    active.set_time(t_start_ms)
    logger.info(f"Restarted from {base} at t={t_start_ms:.3f} ms")

active.post_solve()
write_outputs(t_start_ms)

# ------------------------------------------------------------------ time loop
log = []
k0 = int(round(t_start_ms / DT_MECH_MS))
n_total = int(round(T_END_MS / DT_MECH_MS))
ckpt_every = max(1, int(round(CHECKPOINT_EVERY_MS / DT_MECH_MS)))
output_every = max(1, int(round(OUTPUT_EVERY_MS / DT_MECH_MS)))

for k in range(k0 + 1, n_total + 1):
    t_ms = k * DT_MECH_MS
    wall0 = timer.time()
    active.set_time(t_ms)
    phases_before = {m: s.phase for m, s in states.items()}
    try:
        out = controller.step(t_ms / 1e3, DT_MECH_MS / 1e3)
    except Exception as ex:
        logger.error(f"Mechanics failed at t={t_ms:.1f} ms "
                     f"(LV {PHASE_NAMES[phases_before['LV']]}, RV {PHASE_NAMES[phases_before['RV']]}): {ex}")
        break
    active.post_solve()

    nit = problem.problem.solver.getIterationNumber()
    Jmin, Jmax, ninv = detF_stats()
    ta_myo = active.Ta.x.array[myo_nodes] / 1e3
    lam_myo = active.lmbda.x.array[myo_dofs]
    lv, rv = out["LV"], out["RV"]
    wall = timer.time() - wall0
    log.append([t_ms, ta_myo.max(), lam_myo.min(), lam_myo.max(),
                lv["phase"], lv["V"] * 1e6, lv["P"] / 1e3, lv["P_art"] / 1e3,
                rv["phase"], rv["V"] * 1e6, rv["P"] / 1e3, rv["P_art"] / 1e3,
                nit, Jmin, Jmax, ninv, np.nan, np.nan, 0.0, wall,
                lv["Q"] * 1e6, rv["Q"] * 1e6])
    if k % output_every == 0:
        write_outputs(t_ms)

    logger.info(
        f"t={t_ms:6.1f} ms | Ta max {ta_myo.max():6.1f} kPa, lambda [{lam_myo.min():.3f}, {lam_myo.max():.3f}] | "
        f"LV {PHASE_NAMES[lv['phase']]:8s} V={lv['V'] * 1e6:6.1f} P={lv['P'] / 1e3:6.2f} Q={lv['Q'] * 1e6:6.1f} | "
        f"RV {PHASE_NAMES[rv['phase']]:8s} V={rv['V'] * 1e6:6.1f} P={rv['P'] / 1e3:6.2f} Q={rv['Q'] * 1e6:6.1f} | "
        f"its={nit} J=[{Jmin:.3f}, {Jmax:.3f}] inv={ninv} | {wall:.1f} s")

    transition = any(out[m]["phase"] != phases_before[m] for m in states)
    if k % ckpt_every == 0 or transition:
        write_checkpoint(t_ms)
    if (k - k0) % PLOT_EVERY == 0:
        save_log(np.array(log))

vtx_all.close()
if log:
    arr = np.array(log)
    save_log(arr)
    save_metrics(arr)
    logger.info(f"Finished at t={log[-1][0]:.1f} ms; output in {OUTDIR}")