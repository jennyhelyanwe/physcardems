# run_em_lvonly.py
# Coupled electromechanics for LV-only cases (e.g. the teaching LV ellipsoid).
# Same physics as run_em.py: monodomain ToR-ORd-Land (EP half of the zeta split) driven
# by a prescribed activation map, Land zeta states in UFL, dynamic mechanics, and the
# five-phase cycle controller with a 3-element Windkessel, here for one cavity (LV).
# Boundary conditions ([mechanics.boundary] in the config): epicardial normal spring and damping,
# a weak in-plane basal spring (removes rotation about the long axis), and the base either held in
# its normal direction (base = "fixed_normal", default) or free (base = "free"). With a free base the
# cavity volume is measured from the current basal plane (mean basal displacement, lagged one step).
# Optional: sigmoidal Tref ([land.tref_sigmoid]) so a large systolic Tref does not scale resting
# tension; filling mode ([circulation.lv] filling_mode = "volume" | "pressure" | "gain").
# Outputs (in runs/<case>/<config name>/): log.csv (columns in physcardems.plotting_lv),
# summary.png, metrics.json (biomarkers per beat), pseudo_ecg.csv/.png, simulation.bp
# (open in ParaView), checkpoints/. Serial only.
#
# Usage: python3 scripts/run_em_lvonly.py configs/lv_ellipsoid/baseline_h4.toml
from pathlib import Path
import argparse
import dataclasses
import importlib.util
import json
import logging
import time as timer

import numpy as np
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
from physcardems.active import ZetaSplitConstDt, ZetaSplitSigmoidTref
from physcardems import parameters as mp
from physcardems import ecg as pseudo_ecg
from physcardems.config import load_config, save_config
from physcardems.plotting_lv import log_from_rows, save_log, plot_summary, beat_biomarkers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("em")
comm = MPI.COMM_WORLD
assert comm.size == 1, "serial only for now"

# ------------------------------------------------------------------ settings (from config)
ap = argparse.ArgumentParser(description="Coupled electromechanics simulation")
ap.add_argument("config", help="simulation configuration (.toml or .json)")
args = ap.parse_args()
cfg, case = load_config(args.config)

# case: geometry, data files, tags
DATA = case.root
GEODIR = case.path("geometry", "dir")
ELECTRODES = case.path("ecg", "electrodes")
ELECTRODE_UNIT_SCALE = case.get("ecg", "unit_scale")
TAGS = case.get("geometry", "tags")
MYOCARDIUM_TAGS = tuple(case.get("geometry", "myocardium_cell_tags"))
assert list(case.get("geometry", "cavities", default=[])) == ["LV"], \
    "run_em_lvonly.py needs an LV-only case ([geometry] cavities = [\"LV\"]); use run_em.py for BiV"
BASE_NORMAL_AXIS = case.get("geometry", "base_normal_axis")  # basal plane normal axis (0 = x), or None
ODEFILE = mp.ODEFILE
OUTDIR = Path(cfg.get("outdir") or Path("runs") / case.name / Path(args.config).stem)

# time stepping and output
t = cfg["time"]
DT_MECH_MS, DT_EP_MS, T_END_MS, PCL_MS = t["dt_mech_ms"], t["dt_ep_ms"], t["t_end_ms"], t["pcl_ms"]
OUTPUT_EVERY_MS, CHECKPOINT_EVERY_MS, PLOT_EVERY = t["output_every_ms"], t["checkpoint_every_ms"], t["plot_every"]
RESTART_FROM = cfg.get("restart", {}).get("from") or None

# electrophysiology
CONDUCTIVITIES = cfg["ep"]["conductivities"]
LAT_SHIFT_MS = cfg["ep"]["lat_shift_ms"]
DT_ECG_MS = cfg["ep"]["dt_ecg_ms"]

# mechanics
m = cfg["mechanics"]
QUAD_DEGREE = m["quad_degree"]
KAPPA_PA = m["kappa_pa"]
VALVE_STIFFNESS_SCALE = m["valve_stiffness_scale"]
FIBRE_COMPRESSION_RESISTANCE = m["fibre_compression_resistance"]
REFERENCE_SCALE = m.get("reference_scale")
REFERENCE_LV_VOLUME_ML = m.get("reference_lv_volume_ml")
PRECONDITIONER_LAG = m["preconditioner_lag"]
MATERIAL_PARAMS = dict(m["material"])
BOUNDARY = m.get("boundary", {})
BASE_MODE = BOUNDARY.get("base", "fixed_normal")   # "fixed_normal" (base held along its normal) or "free"
assert BASE_MODE in ("fixed_normal", "free"), f"unknown [mechanics.boundary] base = {BASE_MODE}"

# cell model and Land parameters (base values in parameters.py, changes in the config)
LAND = mp.land_values(overrides=cfg["land"]["overrides"], scales=cfg["land"]["scales"])
EP_OVERRIDES = cfg["cell"]["ep_overrides"]
EP_SCALES = cfg["cell"]["ep_scales"]
PARAM_HASH = mp.parameter_hash(LAND, EP_OVERRIDES, EP_SCALES)
SSDIR = DATA / f"steady_state_pcl{int(PCL_MS)}_{PARAM_HASH}"

N_EP = int(round(DT_MECH_MS / DT_EP_MS))
assert np.isclose(N_EP * DT_EP_MS, DT_MECH_MS), "dt_mech_ms must be a multiple of dt_ep_ms"

# unit conversions
KPA_MS_PER_ML = 1e3 * 1e-3 / 1e-6   # Pa s / m^3
ML_PER_KPA = 1e-6 / 1e3             # m^3 / Pa
ML_PER_MS = 1e-6 / 1e-3             # m^3 / s
MMHG_S_PER_ML = 133.322 / 1e-6
ML_PER_MMHG = 1e-6 / 133.322


def cycle_params(c):
    """[circulation.lv] -> CycleParams. Filling: filling_mode = "volume" (prescribed inflow,
    filling_rate_ml_per_ms), "pressure" (held at the filling pressure, refills as the tissue relaxes),
    or "gain" (Alya-style relaxation law, gain_relaxation_kpa_per_ml and gain_relaxation_kpa_ms_per_ml)."""
    wk = c["windkessel"]
    mode = c.get("filling_mode", "volume")
    assert mode in ("volume", "pressure", "gain"), f"unknown filling_mode {mode}"
    return CycleParams(
        t_zero=c["t_zero_s"], preload_pressure=c["preload_pressure_pa"],
        t_end_diastole=c["t_end_diastole_s"], p_end_diastole=c["p_end_diastole_pa"],
        p_fill=c["p_fill_pa"], period=PCL_MS / 1e3,
        windkessel=WindkesselParams(
            p_init=wk["p_init_pa"],
            resistance=wk["resistance_mmhg_s_per_ml"] * MMHG_S_PER_ML,
            compliance=wk["compliance_ml_per_mmhg"] * ML_PER_MMHG,
            characteristic_impedance=wk["characteristic_impedance_mmhg_s_per_ml"] * MMHG_S_PER_ML),
        filling_rate=c["filling_rate_ml_per_ms"] * ML_PER_MS if mode == "volume" else None,
        filling_gain=(mode == "gain"),
        gain_relaxation=((c.get("gain_relaxation_kpa_per_ml", 0.0) * 1e9,       # kPa/mL -> Pa/m^3
                          c.get("gain_relaxation_kpa_ms_per_ml", 0.0) * 1e6)      # kPa ms/mL -> Pa s/m^3
                         if mode == "gain" else (0.0, 0.0)),
    )


lv_params = cycle_params(cfg["circulation"]["lv"])

OUTDIR.mkdir(parents=True, exist_ok=True)
save_config(cfg, OUTDIR / "config_used.json", param_hash=PARAM_HASH, case=str(case.root))
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
assert geo.cfun is not None, f"geometry in {GEODIR} has no cell tags"


def cavity_volume_ml(marker):
    X = ufl.SpatialCoordinate(geo.mesh)
    N = ufl.FacetNormal(geo.mesh)
    ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)
    form = dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(X, N) * ds(TAGS[marker]))
    return dolfinx.fem.assemble_scalar(form) * 1e6


assert (REFERENCE_SCALE is None) != (REFERENCE_LV_VOLUME_ML is None), \
    "set exactly one of REFERENCE_SCALE or REFERENCE_LV_VOLUME_ML"
V_ref_ml = cavity_volume_ml("LV")
ref_scale = (REFERENCE_SCALE if REFERENCE_SCALE is not None
             else (REFERENCE_LV_VOLUME_ML / V_ref_ml) ** (1.0 / 3.0))
xg = geo.mesh.geometry.x
centroid = xg.mean(axis=0)
if BASE_NORMAL_AXIS is not None:
    centroid[BASE_NORMAL_AXIS] = 0.0  # scale about a point on the basal plane so it stays at 0
xg[:] = centroid + ref_scale * (xg - centroid)
logger.info(f"Reference scaling {ref_scale:.4f}: LV {V_ref_ml:.1f} -> {cavity_volume_ml('LV'):.1f} mL")

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": QUAD_DEGREE})
mesh = geometry.mesh

# With a free base the basal cap leaves its reference plane, so measure the cavity volume from the
# current basal plane: shift the origin of -1/3 (x + u).n to the mean base position, updated each
# step from the last converged state. Must be set before any form is built.
volume_shift = dolfinx.fem.Constant(mesh, np.zeros(3, dtype=dolfinx.default_scalar_type))
if BASE_MODE == "free":
    assert BASE_NORMAL_AXIS is not None, "a free base needs [geometry] base_normal_axis in the case"
    _X, _N = ufl.SpatialCoordinate(mesh), ufl.FacetNormal(mesh)

    def _volume_form(u=None):
        if u is None:
            return (-1.0 / 3.0) * ufl.dot(_X - volume_shift, _N)
        F = ufl.Identity(3) + ufl.grad(u)
        return (-1.0 / 3.0) * ufl.det(F) * ufl.dot(_X + u - volume_shift, ufl.inv(F).T * _N)

    geometry.volume_form = _volume_form
logger.info(f"Base: {BASE_MODE}")

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
applied = mp.apply_to_ode_parameters(P, ep, LAND, EP_OVERRIDES, EP_SCALES)
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
TREF_SIGMOID = cfg["land"].get("tref_sigmoid")
zeta_kwargs = dict(f0=geo.f0, s0=geo.s0, n0=geo.n0, mesh=mesh, dt_ms=DT_MECH_MS, parameters=mech_land)
if TREF_SIGMOID:
    tref_rest = LAND["Tref"] / cfg["land"]["scales"].get("Tref", 1.0) * TREF_SIGMOID["rest_scale"]
    zeta = ZetaSplitSigmoidTref(tref_rest=tref_rest, n_half=TREF_SIGMOID["n_half"],
                                width=TREF_SIGMOID["width"], **zeta_kwargs)
    logger.info(f"Sigmoidal Tref: {tref_rest:.0f} kPa at rest -> {mech_land['Tref']:.0f} kPa when active "
                f"(n_half {TREF_SIGMOID['n_half']}, width {TREF_SIGMOID['width']})")
else:
    zeta = ZetaSplitConstDt(**zeta_kwargs)
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
def _const(v):
    return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(v))


alpha_epi = pulse.Variable(_const(BOUNDARY.get("epi_spring_pa_per_m", 1e8)), "Pa / m")
beta_epi = pulse.Variable(_const(BOUNDARY.get("epi_damping_pa_s_per_m", 5e3)), "Pa s/ m")
robin = [pulse.RobinBC(value=alpha_epi, marker=TAGS["EPI"]),
         pulse.RobinBC(value=beta_epi, marker=TAGS["EPI"], damping=True)]
k_base = BOUNDARY.get("base_tangential_spring_pa_per_m", 0.0)
if k_base > 0.0:  # in-plane basal springs remove the free rotation about the long axis
    robin.append(pulse.RobinBC(value=pulse.Variable(_const(k_base), "Pa / m"),
                               marker=TAGS["BASE"], perpendicular=True))
dirichlet = []
if BASE_NORMAL_AXIS is not None and BASE_MODE == "fixed_normal":
    base_facets = geometry.facet_tags.find(TAGS["BASE"])

    def base_normal_bc(V):
        Vi = V.sub(BASE_NORMAL_AXIS)
        dofs = dolfinx.fem.locate_dofs_topological(Vi, mesh.topology.dim - 1, base_facets)
        return [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, Vi)]

    dirichlet.append(base_normal_bc)
bcs = pulse.BoundaryConditions(robin=tuple(robin), dirichlet=tuple(dirichlet))
cavities = [pulse.problem.Cavity(marker="LV", volume=_const(0.0))]
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
states = {"LV": CavityState(name="LV", params=lv_params)}
controller = BiVCycleController(problem, monitor, states, restore_lag=PRECONDITIONER_LAG)
logger.info("Mechanics problem ready")

# basal displacement along the long axis: mean (sets the volume origin for a free base) and spread
_ds_base = ufl.Measure("ds", domain=mesh, subdomain_data=geometry.facet_tags)(TAGS["BASE"])
_base_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * _ds_base))
_ax = BASE_NORMAL_AXIS if BASE_NORMAL_AXIS is not None else 0
_base_u = dolfinx.fem.form(problem.u[_ax] * _ds_base)
_base_u2 = dolfinx.fem.form(problem.u[_ax] ** 2 * _ds_base)


def base_motion():
    """Mean and standard deviation of the basal displacement along the long axis (m)."""
    mean = dolfinx.fem.assemble_scalar(_base_u) / _base_area
    var = max(dolfinx.fem.assemble_scalar(_base_u2) / _base_area - mean**2, 0.0)
    return mean, var ** 0.5

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
log_rows = []


def save_summary():
    log = log_from_rows(log_rows)
    save_log(log, OUTDIR / "log.csv")
    plot_summary(log, OUTDIR / "summary.png",
                 title=f"{case.name} / {Path(args.config).stem}, parameter hash {PARAM_HASH}",
                 ecg_t=ecg.times, ecg_v=ecg.values, ecg_leads=list(ecg.leads))
    return log


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
    base_dx, base_sd = base_motion()                 # from the last converged step
    if BASE_MODE == "free":
        volume_shift.value[_ax] = base_dx
    phases_before = {m: s.phase for m, s in states.items()}
    try:
        out = controller.step(t_ms / 1e3, DT_MECH_MS / 1e3)
    except Exception as ex:
        logger.error(f"Mechanics failed at t={t_ms:.1f} ms (LV {PHASE_NAMES[phases_before['LV']]}): {ex}")
        break
    zeta.post_solve()
    mech_to_ep()

    nit = problem.problem.solver.getIterationNumber()
    Jmin, Jmax, ninv = detF_stats()
    ta_myo = zeta.Ta_current.x.array[myo_dofs] / 1e3
    lam_myo = zeta.lmbda.x.array[myo_dofs]
    v = ode.values[iv]
    lv = out["LV"]
    wall = timer.time() - wall0
    log_rows.append([t_ms, ta_myo.max(), lam_myo.min(), lam_myo.max(),
                     lv["phase"], lv["V"] * 1e6, lv["P"] / 1e3, lv["P_art"] / 1e3, lv["Q"] * 1e6,
                     nit, Jmin, Jmax, ninv, v.min(), v.max(), wall_ep, wall])
    if k % output_every == 0:
        write_outputs(t_ms)

    logger.info(
        f"t={t_ms:6.1f} ms | Ta max {ta_myo.max():6.1f} kPa, lambda [{lam_myo.min():.3f}, {lam_myo.max():.3f}] | "
        f"LV {PHASE_NAMES[lv['phase']]:8s} V={lv['V'] * 1e6:6.1f} mL P={lv['P'] / 1e3:6.2f} kPa "
        f"Q={lv['Q'] * 1e6:6.1f} mL/s | base dx {base_dx * 1e3:+5.2f} mm (sd {base_sd * 1e3:.2f}) | "
        f"its={nit} J=[{Jmin:.3f}, {Jmax:.3f}] inv={ninv} | EP {wall_ep:.1f} s, total {wall:.1f} s")

    transition = any(out[m]["phase"] != phases_before[m] for m in states)
    if k % ckpt_every == 0 or transition:
        write_checkpoint(t_ms)
    if (k - k0) % PLOT_EVERY == 0:
        save_summary()
    if (k - k0) % ecg_file_every == 0:
        ecg.save_csv(OUTDIR / "pseudo_ecg.csv")
        ecg.plot(OUTDIR / "pseudo_ecg.png")
        save_active_stats()

vtx_all.close()
if log_rows:
    log = save_summary()
    beats = beat_biomarkers(log)
    (OUTDIR / "metrics.json").write_text(json.dumps(
        {"t_reached_ms": float(log["t_ms"][-1]), "t_end_ms": T_END_MS, "beats": beats}, indent=2))
    ecg.save_csv(OUTDIR / "pseudo_ecg.csv")
    ecg.plot(OUTDIR / "pseudo_ecg.png")
    save_active_stats()
    for b in beats:
        logger.info(f"beat {b['beat']}{'' if b['complete'] else ' (incomplete)'}: EF {b['LVEF_pct']:.1f}%, "
                    f"EDV {b['EDV_mL']:.1f} mL, ESV {b['ESV_mL']:.1f} mL, EDP {b['EDP_kPa']:.2f} kPa, "
                    f"Pmax {b['Pmax_kPa']:.1f} kPa")
    logger.info(f"Finished at t={log['t_ms'][-1]:.1f} ms; output in {OUTDIR}")