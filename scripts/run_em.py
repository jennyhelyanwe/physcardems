# run_rodero_em.py
# Coupled electromechanics on rodero_05, EP and mechanics on the same coarse mesh:
# monodomain ToR-ORd-Land (EP half of the zeta split) driven by the Eikonal LAT,
# Land zeta states in UFL, dynamic mechanics with valve plugs, five-phase cycle
# controller. Cell-model parameters come from model_parameters.py. Serial only.
from pathlib import Path
import dataclasses
import importlib.util
import json
import logging
import time as timer

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
import ufl
import io4dolfinx
import gotranx
import beat
import pulse
import cardiac_geometries.geometry

from physcardems.cavity import ControlledCavityDynamicProblem, CavityMonitor
from physcardems.cycle import PHASE_NAMES, CycleParams, WindkesselParams, CavityState, BiVCycleController
from physcardems.active import ZetaSplitConstDt
from physcardems import parameters as mp
from physcardems import ecg as pseudo_ecg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("em")
comm = MPI.COMM_WORLD
assert comm.size == 1, "serial only for now"

# ------------------------------------------------------------------ settings
DATA = Path("/home/shared/rodero_05")
GEODIR = DATA / "rodero_05_dolfinx_v2"
ODEFILE = mp.ODEFILE
ELECTRODES = DATA / "rodero_05_fine_nodefield_electrode_xyz.csv"
# OUTDIR = Path("rodero-em-elife")
OUTDIR = Path("../rodero_05/rodero-em-tref3")

DT_MECH_MS = 2.0
DT_EP_MS = 0.05
T_END_MS = 800.0
PCL_MS = 800.0
LAT_SHIFT_MS = 0.0            # raw LAT starts at 130 ms, just after end-diastole at 120 ms
CONDUCTIVITIES = "Niederer"
ELECTRODE_UNIT_SCALE = 1e-2   # electrode file in cm, mesh in m
DT_ECG_MS = 1.0
QUAD_DEGREE = 4
TAGS = {"LV": 30, "RV": 20, "EPI": 40, "BASE": 10}
VALVE_STIFFNESS_SCALE = 3.0
MYOCARDIUM_TAGS = (1, 2)
PRECONDITIONER_LAG = 20
PLOT_EVERY = 10
OUTPUT_EVERY_MS = 2.0
CHECKPOINT_EVERY_MS = 50.0
RESTART_FROM = None           # e.g. "rodero-em-elife/checkpoints/t_0120.000" (no extension)

# Alya-style unloading: isotropic downscaling of the imaged geometry about its centroid.
# Set exactly one of these.
REFERENCE_SCALE = 0.9
REFERENCE_LV_VOLUME_ML = None

# Passive mechanics
MATERIAL_PARAMS = dict(a=0.61, a_f=1.56, b=7.5, b_f=35.31, a_s=0.7, b_s=33.24, a_fs=0.46, b_fs=5.09)  # moduli in kPa
FIBRE_COMPRESSION_RESISTANCE = False
KAPPA_PA = 1e6


# Cell-model and Land parameters (Margara Land + eLife calibration, see model_parameters.py)
LAND = mp.land_values()
PARAM_HASH = mp.parameter_hash(LAND)
SSDIR = DATA / f"steady_state_pcl{int(PCL_MS)}_{PARAM_HASH}"

N_EP = int(round(DT_MECH_MS / DT_EP_MS))
assert np.isclose(N_EP * DT_EP_MS, DT_MECH_MS), "DT_MECH_MS must be a multiple of DT_EP_MS"

KPA_MS_PER_ML = 1e3 * 1e-3 / 1e-6   # Pa s / m^3
ML_PER_KPA = 1e-6 / 1e3             # m^3 / Pa
ML_PER_MS = 1e-6 / 1e-3             # m^3 / s
MMHG_S_PER_ML = 133.322 / 1e-6
ML_PER_MMHG = 1e-6 / 133.322

lv_params = CycleParams(
    t_zero=0.05, preload_pressure=500.0, t_end_diastole=0.12, p_end_diastole=1000.0,
    p_fill=500.0, period=PCL_MS / 1e3,
    windkessel=WindkesselParams(p_init=9000.0,
                                resistance=1.1 * MMHG_S_PER_ML,
                                compliance=1.5 * ML_PER_MMHG,
                                characteristic_impedance=0.03 * MMHG_S_PER_ML),
    filling_rate=0.046 * ML_PER_MS,
)
rv_params = CycleParams(
    t_zero=0.05, preload_pressure=170.0, t_end_diastole=0.12, p_end_diastole=330.0,
    p_fill=170.0, period=PCL_MS / 1e3,
    windkessel=WindkesselParams(p_init=3000.0,
                                resistance=0.1 * MMHG_S_PER_ML,
                                compliance=4.0 * ML_PER_MMHG,
                                characteristic_impedance=0.01 * MMHG_S_PER_ML),
    filling_rate=0.046 * ML_PER_MS,
)

# # LV: eLife Figure 1F, 2-element (R_c = 0) to reproduce the calibrated model first
# lv_params = CycleParams(
#     t_zero=0.05, preload_pressure=500.0, t_end_diastole=0.12, p_end_diastole=1000.0,
#     p_fill=500.0, period=PCL_MS / 1e3,
#     windkessel=WindkesselParams(p_init=8450.0,
#                                 resistance=21.59 * KPA_MS_PER_ML,
#                                 compliance=1.392 * ML_PER_KPA,
#                                 characteristic_impedance=0.0),
#     ejection_pressure=8450.0,
#     filling_rate=0.046 * ML_PER_MS,
# )
# # RV: not in Figure 1F; previous pulmonary values, same filling rate as the LV
# rv_params = CycleParams(
#     t_zero=0.05, preload_pressure=170.0, t_end_diastole=0.12, p_end_diastole=330.0,
#     p_fill=170.0, period=PCL_MS / 1e3,
#     windkessel=WindkesselParams(p_init=3000.0,
#                                 resistance=0.1 * MMHG_S_PER_ML,
#                                 compliance=4.0 * ML_PER_MMHG,
#                                 characteristic_impedance=0.0),
#     filling_rate=0.046 * ML_PER_MS,
# )

OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTDIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GEN_DIR = OUTDIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)


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
assert geo.cfun is not None, "geometry has no cell tags; use rodero_05_dolfinx_v2"


def cavity_volume_ml(marker):
    X = ufl.SpatialCoordinate(geo.mesh)
    N = ufl.FacetNormal(geo.mesh)
    ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)
    form = dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(X, N) * ds(TAGS[marker]))
    return dolfinx.fem.assemble_scalar(form) * 1e6


assert (REFERENCE_SCALE is None) != (REFERENCE_LV_VOLUME_ML is None), \
    "set exactly one of REFERENCE_SCALE or REFERENCE_LV_VOLUME_ML"
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

# ------------------------------------------------------------------ EP: node fields, cell model, monodomain
W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
fields = {}
for name in ("lat_ms", "celltype", "sf_IKs"):
    fields[name] = dolfinx.fem.Function(W, name=name)
    io4dolfinx.read_function(filename=GEODIR / "ep_node_fields.bp", u=fields[name], name=name)
lat = fields["lat_ms"].x.array.copy() - LAT_SHIFT_MS
ct = np.rint(fields["celltype"].x.array).astype(int)
N = lat.size

ep_file = GEN_DIR / "ep_zetasplit.py"
if not ep_file.exists():
    ode_full = gotranx.load_ode(ODEFILE)
    mech_comp = ode_full.get_component("mechanics")
    mech_ode = mech_comp.to_ode()
    ep_ode = ode_full - mech_comp
    code = gotranx.cli.gotran2py.get_code(ep_ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
                                          missing_values=mech_ode.missing_variables)
    ep_file.write_text(code)
spec = importlib.util.spec_from_file_location("ep_zetasplit", ep_file)
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}
ss = {}
for c, cname in CELLTYPES.items():
    data_ss = json.loads((SSDIR / f"celltype{c}_{cname}.json").read_text())
    assert data_ss.get("parameter_hash") == PARAM_HASH, \
        f"steady state in {SSDIR} was paced with a different parameter set; rerun pace_steady_state.py"
    ss[c] = data_ss["states"]
ep_state_names = sorted(ep.state, key=ep.state.get)
y0 = np.zeros((len(ep_state_names), N))
missing = np.zeros((len(ep.missing), N))
for c in CELLTYPES:
    y0[:, ct == c] = np.array([ss[c][s] for s in ep_state_names])[:, None]
    for name, idx in ep.missing.items():
        missing[idx, ct == c] = ss[c][name]

P = np.repeat(ep.init_parameter_values(i_Stim_Period=PCL_MS, lmbda=1.0, dLambda=0.0)[:, None], N, axis=1)
P[ep.parameter_index("celltype")] = ct.astype(float)
P[ep.parameter_index("i_Stim_Start")] = lat
applied = mp.apply_to_ode_parameters(P, ep, LAND)
logger.info(f"Parameter hash {PARAM_HASH}; EP parameters applied: {applied}")
i_lmbda = ep.parameter_index("lmbda")
iv, ica, iXS, iXW = (ep.state_index(s) for s in ("v", "cai", "XS", "XW"))
iZs, iZw = ep.missing_index("Zetas"), ep.missing_index("Zetaw")

ep_time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
cond = beat.conductivities.default_conductivities(CONDUCTIVITIES)
M = beat.conductivities.define_conductivity_tensor(f0=geo.f0, **cond)
C_m = (1.0 * beat.units.ureg("uF/cm**2")).to("uF/m**2").magnitude
pde = beat.MonodomainModel(time=ep_time, mesh=mesh, M=M, I_s=None, C_m=C_m)
ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(W, name="v_ode"), v_pde=pde.state,
    fun=ep.generalized_rush_larsen, init_states=y0, parameters=P,
    num_states=y0.shape[0], v_index=iv,
    missing_variables=missing, num_missing_variables=missing.shape[0],
)
ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
logger.info(f"EP: {N} nodes, {y0.shape[0]} states, LAT {lat.min():.1f} to {lat.max():.1f} ms, {N_EP} EP steps per mechanics step")

# ------------------------------------------------------------------ mechanics
mech_land = mp.mechanics_parameters(LAND)
logger.info(f"Land (mechanics side): {mech_land}")
zeta = ZetaSplitConstDt(f0=geo.f0, s0=geo.s0, n0=geo.n0, mesh=mesh, dt_ms=DT_MECH_MS, parameters=mech_land)
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
                                         parameters=params, true_u_active=ScaledModel(zeta, myo_mask))
zeta.register(problem.u)
monitor = CavityMonitor(problem)
states = {"LV": CavityState(name="LV", params=lv_params), "RV": CavityState(name="RV", params=rv_params)}
controller = BiVCycleController(problem, monitor, states, restore_lag=PRECONDITIONER_LAG)
logger.info("Mechanics problem ready")

# ------------------------------------------------------------------ EP <-> mechanics transfers (same mesh)
DG1 = zeta.function_space
xs_p1, xw_p1 = dolfinx.fem.Function(W), dolfinx.fem.Function(W)
zs_p1, zw_p1, lam_p1 = dolfinx.fem.Function(W), dolfinx.fem.Function(W), dolfinx.fem.Function(W)
myo_dg1 = dolfinx.fem.Function(DG1)
myo_dg1.interpolate(myo_mask)


def ep_to_mech():
    xs_p1.x.array[:] = ode.values[iXS]
    xw_p1.x.array[:] = ode.values[iXW]
    zeta.XS.interpolate(xs_p1)
    zeta.XW.interpolate(xw_p1)


def mech_to_ep():
    zs_p1.interpolate(zeta._Zetas)
    zw_p1.interpolate(zeta._Zetaw)
    lam_p1.interpolate(zeta.lmbda)
    missing[iZs, :] = zs_p1.x.array
    missing[iZw, :] = zw_p1.x.array
    P[i_lmbda, :] = lam_p1.x.array


zs_p1.x.array[:] = missing[iZs]
zw_p1.x.array[:] = missing[iZw]
for target, src in ((zeta.Zetas_prev, zs_p1), (zeta._Zetas, zs_p1), (zeta.Zetaw_prev, zw_p1), (zeta._Zetaw, zw_p1)):
    target.interpolate(src)

# ------------------------------------------------------------------ diagnostics
X_ref = np.array([[0.25, 0.25, 0.25], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dolfinx.default_real_type)
J_expr = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(problem.u)), X_ref)
local_cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)


def detF_stats():
    J = J_expr.eval(mesh, local_cells)
    return float(J.min()), float(J.max()), int(np.sum(np.min(J, axis=1) <= 0.0))


electrodes = pseudo_ecg.load_electrodes(ELECTRODES, ELECTRODE_UNIT_SCALE)
ecg_points, ecg_leads = pseudo_ecg.standard_leads(electrodes)
ecg = pseudo_ecg.PseudoECG(pde.state, ecg_points, ecg_leads, sigma_b=1.0)
ecg_every = max(1, int(round(DT_ECG_MS / DT_EP_MS)))

# ------------------------------------------------------------------ combined output: all fields on P1 in one .bp
Wv = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
u_out = dolfinx.fem.Function(Wv, name="u")
v_out = dolfinx.fem.Function(W, name="v_mV")
cai_out = dolfinx.fem.Function(W, name="cai_mM")
ct_out = dolfinx.fem.Function(W, name="celltype")
ct_out.x.array[:] = ct
lat_out = dolfinx.fem.Function(W, name="lat_ms")
lat_out.x.array[:] = lat
myo_out = dolfinx.fem.Function(W, name="myocardium")

ACTIVE_FIELD_NAMES = ("lambda", "h_lambda", "active_fraction", "Zetas", "dLambda_per_ms", "Ta_kPa")
active_dg1 = {n: dolfinx.fem.Function(DG1, name=f"{n}_dg1") for n in ACTIVE_FIELD_NAMES}
active_out = {n: dolfinx.fem.Function(W, name=n) for n in ACTIVE_FIELD_NAMES}

# lumped L2 projection onto P1 = volume-weighted average of the surrounding cells
w_test = ufl.TestFunction(W)
dx_out = ufl.Measure("dx", domain=mesh)
lumped_mass = dolfinx.fem.assemble_vector(dolfinx.fem.form(w_test * dx_out)).array.copy()
proj_forms = {n: dolfinx.fem.form(active_dg1[n] * w_test * dx_out) for n in ACTIVE_FIELD_NAMES}
myo_out.x.array[:] = dolfinx.fem.assemble_vector(dolfinx.fem.form(myo_mask * w_test * dx_out)).array / lumped_mass

vtx_all = dolfinx.io.VTXWriter(
    comm, OUTDIR / "simulation.bp",
    [u_out, v_out, cai_out, *active_out.values(), ct_out, lat_out, myo_out], engine="BP4",
)

ct_dg1 = dolfinx.fem.Function(DG1)
ct_dg1.interpolate(ct_out)
BETA0 = zeta._parameters["Beta0"]
active_stats_rows = []
ACTIVE_STATS_HEADER = ("t_ms,Ta_p5_kPa,Ta_p50_kPa,Ta_p95_kPa,frac_h_zero,"
                       "Ta_med_endo,lambda_med_endo,Ta_med_epi,lambda_med_epi,Ta_med_mid,lambda_med_mid")


def write_outputs(t_ms):
    lam = zeta.lmbda.x.array
    lam_c = np.minimum(1.2, lam)
    h = np.maximum(0.0, 1.0 + BETA0 * (lam_c + np.minimum(lam_c, 0.87) - 1.87))
    af = np.maximum(0.0, zeta.XS.x.array * (zeta._Zetas.x.array + 1.0) + zeta.XW.x.array * zeta._Zetaw.x.array)
    ta_kpa = zeta.Ta_current.x.array * myo_dg1.x.array / 1e3
    active_dg1["lambda"].x.array[:] = lam
    active_dg1["h_lambda"].x.array[:] = h
    active_dg1["active_fraction"].x.array[:] = af
    active_dg1["Zetas"].x.array[:] = zeta._Zetas.x.array
    active_dg1["dLambda_per_ms"].x.array[:] = zeta._dLambda.x.array
    active_dg1["Ta_kPa"].x.array[:] = ta_kpa
    for n in ACTIVE_FIELD_NAMES:
        active_out[n].x.array[:] = dolfinx.fem.assemble_vector(proj_forms[n]).array / lumped_mass

    u_out.interpolate(problem.u)
    v_out.x.array[:] = ode.values[iv]
    cai_out.x.array[:] = ode.values[ica]
    vtx_all.write(t_ms)

    myo = myo_dg1.x.array > 0.5
    ct_d = np.rint(ct_dg1.x.array).astype(int)
    row = [t_ms, *np.percentile(ta_kpa[myo], [5, 50, 95]), float(np.mean(h[myo] == 0.0))]
    for c in (0, 1, 2):
        sel = myo & (ct_d == c)
        row += [float(np.median(ta_kpa[sel])), float(np.median(lam[sel]))]
    active_stats_rows.append(row)


def save_active_stats():
    if active_stats_rows:
        np.savetxt(OUTDIR / "active_stats.csv", np.array(active_stats_rows), delimiter=",",
                   header=ACTIVE_STATS_HEADER, comments="")


# ------------------------------------------------------------------ checkpoints
ZETA_FIELDS = {"XS": "XS", "XW": "XW", "Zetas": "_Zetas", "Zetaw": "_Zetaw", "Zetas_prev": "Zetas_prev",
               "Zetaw_prev": "Zetaw_prev", "lmbda": "lmbda", "lmbda_prev": "lmbda_prev",
               "dLambda": "_dLambda", "Ta": "Ta_current"}


def write_checkpoint(t_ms):
    base = CKPT_DIR / f"t_{t_ms:08.3f}"
    for i, (name, f) in enumerate((("u", problem.u), ("v", problem.v_old), ("a", problem.a_old))):
        mode = io4dolfinx.FileMode.write if i == 0 else io4dolfinx.FileMode.append
        io4dolfinx.write_function_on_input_mesh(f"{base}.bp", f, time=0.0, name=name, mode=mode, backend="adios2")
    np.savez(f"{base}_ep.npz", ode_values=ode.values, missing=missing, lmbda_row=P[i_lmbda],
             **{k: getattr(zeta, a).x.array for k, a in ZETA_FIELDS.items()})
    data = {
        "t_ms": t_ms, "dt_mech_ms": DT_MECH_MS, "dt_ep_ms": DT_EP_MS,
        "reference_scale": ref_scale, "parameter_hash": PARAM_HASH,
        "pressures": {m: monitor.pressure(m) for m in states},
        "states": {m: {k: v for k, v in dataclasses.asdict(s).items() if k not in ("name", "params")}
                   for m, s in states.items()},
    }
    Path(f"{base}.json").write_text(json.dumps(data, indent=2))
    logger.info(f"Checkpoint written: {base}")


# ------------------------------------------------------------------ start or restart
if RESTART_FROM is None:
    ep_to_mech()
    zeta.t.value = 0.0
    problem.solve()
    controller.initialize(0.0, {m: monitor.volume(m) for m in states})
    t_start_ms = 0.0
else:
    base = str(RESTART_FROM)
    data = json.loads(Path(f"{base}.json").read_text())
    assert np.isclose(data["dt_mech_ms"], DT_MECH_MS) and np.isclose(data["dt_ep_ms"], DT_EP_MS)
    assert np.isclose(data.get("reference_scale", 1.0), ref_scale), "checkpoint used a different reference scaling"
    if data.get("parameter_hash") != PARAM_HASH:
        logger.warning("Checkpoint parameter hash differs; valid only if the change cannot have acted yet")
    for name, f in (("u", problem.u), ("v", problem.v_old), ("a", problem.a_old)):
        io4dolfinx.read_function(f"{base}.bp", f, time=0.0, name=name, backend="adios2")
        f.x.scatter_forward()
    problem.u_old.x.array[:] = problem.u.x.array
    for m, s in states.items():
        for k, v in data["states"][m].items():
            setattr(s, k, v)
        monitor.init_pressure(m, data["pressures"][m])
    saved = np.load(f"{base}_ep.npz")
    ode.values[:] = saved["ode_values"]
    missing[:] = saved["missing"]
    P[i_lmbda] = saved["lmbda_row"]
    for k, a in ZETA_FIELDS.items():
        getattr(zeta, a).x.array[:] = saved[k]
    ode.to_dolfin()
    ode.ode_to_pde()
    pde.assign_previous()
    t_start_ms = data["t_ms"]
    zeta.t.value = t_start_ms
    zeta._t_prev.value = t_start_ms
    logger.info(f"Restarted from {base} at t={t_start_ms:.3f} ms")

write_outputs(t_start_ms)
ecg.compute(t_start_ms)

# ------------------------------------------------------------------ time loop
header = ("t_ms,Ta_max_kPa,lmbda_min,lmbda_max,phase_lv,V_lv_mL,P_lv_kPa,Pc_lv_kPa,"
          "phase_rv,V_rv_mL,P_rv_kPa,Pc_rv_kPa,newton_its,J_min,J_max,n_inverted,v_min,v_max,"
          "wall_ep_s,wall_s,Q_lv_mL_s,Q_rv_mL_s")
log = []


LEAD_ROWS = [["I", "II", "III", "aVR", "aVL", "aVF"], ["V1", "V2", "V3", "V4", "V5", "V6"]]

def save_plots(arr):
    np.savetxt(OUTDIR / "log.csv", arr, delimiter=",", header=header, comments="")
    t = arr[:, 0]
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 12, height_ratios=[1.0, 1.0, 0.55, 0.55], hspace=0.55, wspace=0.9)
    ax = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            ax[r, c] = fig.add_subplot(gs[r, 4 * c:4 * c + 4])

    ax[0, 0].plot(t, arr[:, 6], label="LV")
    ax[0, 0].plot(t, arr[:, 10], label="RV")
    ax[0, 0].plot(t, arr[:, 7], "--", label="P_c (sys)")
    ax[0, 0].plot(t, arr[:, 11], "--", label="P_c (pul)")
    ax[0, 0].set_title("Pressure (kPa)")
    ax[0, 1].plot(t, arr[:, 5], label="LV volume")
    ax[0, 1].plot(t, arr[:, 9], label="RV volume")
    axq = ax[0, 1].twinx()
    axq.plot(t, arr[:, 20], "C0:", label="LV outflow")
    axq.plot(t, arr[:, 21], "C1:", label="RV outflow")
    axq.set_ylabel("outflow (mL/s)")
    axq.legend(loc="lower right")
    ax[0, 1].set_title("Volume (mL) and outflow (mL/s)")
    ax[0, 2].plot(arr[:, 5], arr[:, 6], label="LV")
    ax[0, 2].plot(arr[:, 9], arr[:, 10], label="RV")
    ax[0, 2].set_title("PV loop (kPa vs mL)")
    ax[1, 0].plot(t, arr[:, 1], "k", label="max Ta, myocardium (kPa)")
    axp = ax[1, 0].twinx()
    axp.step(t, arr[:, 4], where="post", label="LV phase")
    axp.step(t, arr[:, 8], where="post", label="RV phase")
    axp.set_yticks(list(PHASE_NAMES))
    axp.set_yticklabels(list(PHASE_NAMES.values()))
    ax[1, 0].set_title("Active tension and phase")
    ax[1, 1].plot(t, arr[:, 13], label="J min")
    ax[1, 1].plot(t, arr[:, 14], label="J max")
    ax[1, 1].set_title("det F")
    ax[1, 2].plot(t, arr[:, 2], label="lambda min")
    ax[1, 2].plot(t, arr[:, 3], label="lambda max")
    ax[1, 2].set_title("Fibre stretch, myocardium")
    for a in ax.flat:
        a.legend(loc="best")
        a.set_xlabel("t (ms)")
    ax[0, 2].set_xlabel("volume (mL)")

    if len(ecg.times) > 1:
        te = np.array(ecg.times)
        ve = np.array(ecg.values)
        col = {name: i for i, name in enumerate(ecg.leads)}
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

    fig.suptitle(f"parameter hash {PARAM_HASH}, reference scale {ref_scale:.3f}", fontsize=13)
    fig.savefig(OUTDIR / "summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

k0 = int(round(t_start_ms / DT_MECH_MS))
n_total = int(round(T_END_MS / DT_MECH_MS))
k_ep = k0 * N_EP
ckpt_every = max(1, int(round(CHECKPOINT_EVERY_MS / DT_MECH_MS)))
output_every = max(1, int(round(OUTPUT_EVERY_MS / DT_MECH_MS)))
ecg_file_every = max(1, int(round(50.0 / DT_MECH_MS)))
myo_dofs = myo_dg1.x.array > 0.5

for k in range(k0 + 1, n_total + 1):
    t_ms = k * DT_MECH_MS
    wall0 = timer.time()

    for _ in range(N_EP):
        t0 = k_ep * DT_EP_MS
        k_ep += 1
        t1 = k_ep * DT_EP_MS
        ep_solver.step((t0, t1))
        if k_ep % ecg_every == 0:
            ecg.compute(t1)
    wall_ep = timer.time() - wall0

    ep_to_mech()
    zeta.t.value = t_ms
    phases_before = {m: s.phase for m, s in states.items()}
    try:
        out = controller.step(t_ms / 1e3, DT_MECH_MS / 1e3)
    except Exception as ex:
        logger.error(f"Mechanics failed at t={t_ms:.1f} ms "
                     f"(LV {PHASE_NAMES[phases_before['LV']]}, RV {PHASE_NAMES[phases_before['RV']]}): {ex}")
        break
    zeta.post_solve()
    mech_to_ep()

    nit = problem.problem.solver.getIterationNumber()
    Jmin, Jmax, ninv = detF_stats()
    ta_myo = zeta.Ta_current.x.array[myo_dofs] / 1e3
    lam_myo = zeta.lmbda.x.array[myo_dofs]
    v = ode.values[iv]
    lv, rv = out["LV"], out["RV"]
    wall = timer.time() - wall0
    log.append([t_ms, ta_myo.max(), lam_myo.min(), lam_myo.max(),
                lv["phase"], lv["V"] * 1e6, lv["P"] / 1e3, lv["P_art"] / 1e3,
                rv["phase"], rv["V"] * 1e6, rv["P"] / 1e3, rv["P_art"] / 1e3,
                nit, Jmin, Jmax, ninv, v.min(), v.max(), wall_ep, wall,
                lv["Q"] * 1e6, rv["Q"] * 1e6])
    if k % output_every == 0:
        write_outputs(t_ms)

    logger.info(
        f"t={t_ms:6.1f} ms | Ta max {ta_myo.max():6.1f} kPa, lambda [{lam_myo.min():.3f}, {lam_myo.max():.3f}] | "
        f"LV {PHASE_NAMES[lv['phase']]:8s} V={lv['V'] * 1e6:6.1f} P={lv['P'] / 1e3:6.2f} Q={lv['Q'] * 1e6:6.1f} | "
        f"RV {PHASE_NAMES[rv['phase']]:8s} V={rv['V'] * 1e6:6.1f} P={rv['P'] / 1e3:6.2f} Q={rv['Q'] * 1e6:6.1f} | "
        f"its={nit} J=[{Jmin:.3f}, {Jmax:.3f}] inv={ninv} | EP {wall_ep:.1f} s, total {wall:.1f} s")

    transition = any(out[m]["phase"] != phases_before[m] for m in states)
    if k % ckpt_every == 0 or transition:
        write_checkpoint(t_ms)
    if (k - k0) % PLOT_EVERY == 0:
        save_plots(np.array(log))
    if (k - k0) % ecg_file_every == 0:
        ecg.save_csv(OUTDIR / "pseudo_ecg.csv")
        ecg.plot(OUTDIR / "pseudo_ecg.png")
        save_active_stats()

vtx_all.close()
if log:
    save_plots(np.array(log))
    ecg.save_csv(OUTDIR / "pseudo_ecg.csv")
    ecg.plot(OUTDIR / "pseudo_ecg.png")
    save_active_stats()
    logger.info(f"Finished at t={log[-1][0]:.1f} ms; output in {OUTDIR}")