# Dynamic BiV mechanics on rodero_05, adapted from fenicsx-pulse time_dependent_bestel_biv.py
from pathlib import Path
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
import ufl
from scipy.integrate import solve_ivp
import circulation.bestel
import cardiac_geometries.geometry
import pulse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rodero")
comm = MPI.COMM_WORLD

# ------------------------------------------------------------------ settings
GEODIR = Path("../rodero_05/rodero_05_dolfinx")
OUTDIR = Path("../rodero_05/rodero-dynamic")


PRESSURE_MODE = "ramp_active"   # "ramp" | "ramp_active" | "bestel"
ACTIVE = True                   # False holds Ta = 0 in every mode
T_END = 0.8                     # s, ramp modes only
T_ACT_SHIFT = 0.1               # s; Bestel onset (t_sys = 0.16) lands at 0.26 s, after the plateau
SIGMA_0 = 3e4                   # Pa, Bestel contractility (demo value is 1.5e5)
DT = 2e-3                # s
T_RAMP = 0.2             # s
P_LV_TARGET = 1.6e3      # Pa (about 12 mmHg)
P_RV_TARGET = 0.5e3      # Pa (about 4 mmHg)
QUAD_DEGREE = 4
TAGS = {"LV": 30, "RV": 20, "EPI": 40, "BASE": 10}
PLOT_EVERY = 10

OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ geometry
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=GEODIR)
for name, tag in TAGS.items():
    assert int(geo.markers[name][0]) == tag, f"{name}: expected tag {tag}, file has {geo.markers[name]}"

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": QUAD_DEGREE})
mesh = geometry.mesh
V_lv0 = geometry.volume("LV")
V_rv0 = geometry.volume("RV")
if comm.rank == 0:
    logger.info(f"Reference volumes: LV {V_lv0 * 1e6:.1f} mL, RV {V_rv0 * 1e6:.1f} mL")

# ------------------------------------------------------------------ model
material_params = pulse.HolzapfelOgden.orthotropic_parameters()
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore
Ta = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")
active_model = pulse.ActiveStress(geo.f0, activation=Ta)
comp_model = pulse.compressibility.Compressible2()
viscoelastic_model = pulse.viscoelasticity.Viscous()
model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
    viscoelasticity=viscoelastic_model,
)

# ------------------------------------------------------------------ boundary conditions
traction_lv = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")
traction_rv = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")
neumann_lv = pulse.NeumannBC(traction=traction_lv, marker=TAGS["LV"])
neumann_rv = pulse.NeumannBC(traction=traction_rv, marker=TAGS["RV"])

alpha_epi = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1e8)), "Pa / m")
beta_epi = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m")
robin_epi_u = pulse.RobinBC(value=alpha_epi, marker=TAGS["EPI"])
robin_epi_v = pulse.RobinBC(value=beta_epi, marker=TAGS["EPI"], damping=True)
# BASE (tag 10): no boundary condition

bcs = pulse.BoundaryConditions(neumann=(neumann_lv, neumann_rv), robin=(robin_epi_u, robin_epi_v))

params = pulse.problem.DynamicProblem.default_parameters()
params["dt"] = pulse.Variable(DT, "s")
# params["petsc_options"] = {**params["petsc_options"], "snes_monitor": None}
params["petsc_options"] = {
    **params["petsc_options"],
    "snes_monitor": None,
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "ksp_max_it": 100,
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-14,
    "ksp_converged_reason": None,
    "snes_lag_preconditioner": 20,
    "snes_lag_preconditioner_persists": True,
}
problem = pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs, parameters=params)
problem.solve()

# ------------------------------------------------------------------ load traces
if PRESSURE_MODE in ("ramp", "ramp_active"):
    times = DT * np.arange(1, int(round(T_END / DT)) + 1)
    w = 0.5 * (1.0 - np.cos(np.pi * np.clip(times / T_RAMP, 0.0, 1.0)))
    lv_pressure = P_LV_TARGET * w
    rv_pressure = P_RV_TARGET * w
    activation = np.zeros_like(times)
    if PRESSURE_MODE == "ramp_active" and ACTIVE:
        t_act = times - T_ACT_SHIFT
        mask = t_act >= 0.0
        act_model = circulation.bestel.BestelActivation(parameters={"sigma_0": SIGMA_0})
        sol = solve_ivp(act_model, [0.0, t_act[mask][-1]], [0.0], t_eval=t_act[mask], method="Radau")
        activation[mask] = sol.y[0]
elif PRESSURE_MODE == "bestel":
    if not ACTIVE and comm.rank == 0:
        logger.warning("Bestel pressures reach about 16 kPa; without active tension the tissue will overinflate")
    times = DT * np.arange(1, int(round(1.0 / DT)) + 1)
    t_span = [0.0, times[-1]]
    lv_model = circulation.bestel.BestelPressure(parameters=dict(
        t_sys_pre=0.17, t_dias_pre=0.484, gamma=0.005, a_max=5.0, a_min=-30.0,
        alpha_pre=5.0, alpha_mid=15.0, sigma_pre=12000.0, sigma_mid=16000.0))
    rv_model = circulation.bestel.BestelPressure(parameters=dict(
        t_sys_pre=0.17, t_dias_pre=0.484, gamma=0.005, a_max=5.0, a_min=-30.0,
        alpha_pre=1.0, alpha_mid=10.0, sigma_pre=3000.0, sigma_mid=4000.0))
    lv_pressure = solve_ivp(lv_model, t_span, [0.0], t_eval=times, method="Radau").y[0]
    rv_pressure = solve_ivp(rv_model, t_span, [0.0], t_eval=times, method="Radau").y[0]
    activation = np.zeros_like(times)
    if ACTIVE:
        activation = solve_ivp(circulation.bestel.BestelActivation(), t_span, [0.0],
                               t_eval=times, method="Radau").y[0]
else:
    raise ValueError(PRESSURE_MODE)

# ------------------------------------------------------------------ diagnostics
X_ref = np.array([[0.25, 0.25, 0.25], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 dtype=dolfinx.default_real_type)
J_expr = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(problem.u)), X_ref)
local_cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)


def detF_stats():
    J = J_expr.eval(mesh, local_cells)
    n_inv = int(np.sum(np.min(J, axis=1) <= 0.0))
    return (comm.allreduce(float(J.min()), op=MPI.MIN),
            comm.allreduce(float(J.max()), op=MPI.MAX),
            comm.allreduce(n_inv, op=MPI.SUM))


volume_form = geometry.volume_form(u=problem.u)
lv_volume_form = dolfinx.fem.form(volume_form * geometry.ds(TAGS["LV"]))
rv_volume_form = dolfinx.fem.form(volume_form * geometry.ds(TAGS["RV"]))


def volume(form):
    return comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


# ------------------------------------------------------------------ time loop
vtx = dolfinx.io.VTXWriter(comm, OUTDIR / "displacement.bp", [problem.u], engine="BP4")
vtx.write(0.0)

log = []
header = "t,p_lv_Pa,p_rv_Pa,Ta_Pa,V_lv_mL,V_rv_mL,newton_its,J_min,J_max,n_inverted"

for i, (t, plv, prv, tai) in enumerate(zip(times, lv_pressure, rv_pressure, activation)):
    traction_lv.assign(plv)
    traction_rv.assign(prv)
    Ta.assign(tai)
    try:
        problem.solve()
    except Exception as ex:
        if comm.rank == 0:
            logger.error(f"Solve failed at t={t:.4f} s (p_lv={plv:.1f} Pa, p_rv={prv:.1f} Pa): {ex}")
        break

    nit = problem.problem.solver.getIterationNumber()
    Jmin, Jmax, ninv = detF_stats()
    Vlv, Vrv = volume(lv_volume_form) * 1e6, volume(rv_volume_form) * 1e6
    log.append([t, plv, prv, tai, Vlv, Vrv, nit, Jmin, Jmax, ninv])
    vtx.write(t)

    if comm.rank == 0:
        logger.info(f"t={t:.3f} p_lv={plv:7.1f} p_rv={prv:7.1f} Pa | V_lv={Vlv:6.1f} V_rv={Vrv:6.1f} mL "
                    f"| its={nit} | J=[{Jmin:.3f}, {Jmax:.3f}] inverted={ninv}")
        if (i + 1) % PLOT_EVERY == 0 or i == len(times) - 1:
            arr = np.array(log)
            np.savetxt(OUTDIR / "log.csv", arr, delimiter=",", header=header, comments="")
            fig, ax = plt.subplots(2, 2, figsize=(11, 8))
            ax[0, 0].plot(arr[:, 0], arr[:, 1] / 1e3, label="LV")
            ax[0, 0].plot(arr[:, 0], arr[:, 2] / 1e3, label="RV")
            ax[0, 0].set_title("Pressure (kPa)")
            ax0b = ax[0, 0].twinx()
            ax0b.plot(arr[:, 0], arr[:, 3] / 1e3, "k--", label="Ta (kPa)")
            ax[0, 1].plot(arr[:, 0], arr[:, 4], label="LV")
            ax[0, 1].plot(arr[:, 0], arr[:, 5], label="RV")
            ax[0, 1].set_title("Volume (mL)")
            ax[1, 0].plot(arr[:, 4], arr[:, 1] / 1e3, label="LV")
            ax[1, 0].plot(arr[:, 5], arr[:, 2] / 1e3, label="RV")
            ax[1, 0].set_title("PV (kPa vs mL)")
            ax[1, 1].plot(arr[:, 0], arr[:, 7], label="J min")
            ax[1, 1].plot(arr[:, 0], arr[:, 8], label="J max")
            ax[1, 1].set_title("det F")
            for a in ax.flat:
                a.legend()
            fig.tight_layout()
            fig.savefig(OUTDIR / "summary.png")
            plt.close(fig)

vtx.close()
if comm.rank == 0 and log:
    np.savetxt(OUTDIR / "log.csv", np.array(log), delimiter=",", header=header, comments="")
    logger.info(f"Finished {len(log)} of {len(times)} steps; output in {OUTDIR}")