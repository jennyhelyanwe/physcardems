# run_rodero_cycle.py
# Dynamic BiV mechanics on rodero_05 driven by the five-phase cycle controller.
from pathlib import Path
import dataclasses
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
import ufl
from scipy.integrate import solve_ivp
import io4dolfinx
import circulation.bestel
import cardiac_geometries.geometry
import pulse

from cavity_control import ControlledCavityDynamicProblem, CavityMonitor
from cycle_controller import (Phase, PHASE_NAMES, CycleParams, WindkesselParams,
                              CavityState, BiVCycleController)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cycle")
comm = MPI.COMM_WORLD

# ------------------------------------------------------------------ settings
GEODIR = Path("/home/shared/rodero_05/rodero_05_dolfinx")
# OUTDIR = Path("rodero-cycle")
OUTDIR = Path("rodero-cycle-restart")

DT = 2e-3                  # s
T_END = 0.8                # s
QUAD_DEGREE = 4
TAGS = {"LV": 30, "RV": 20, "EPI": 40, "BASE": 10}
PLOT_EVERY = 10
PRECONDITIONER_LAG = 20

SIGMA_0 = 1.5e5            # Pa, Bestel contractility (peak Ta about 0.79 * SIGMA_0)
ACT_ONSET = 0.12           # s, aligned with end-diastole
ACT_DURATION = 0.324       # s, Bestel t_dias - t_sys

CHECKPOINT_EVERY = 0.05    # s, plus a checkpoint at every phase transition
# RESTART_FROM = None        # e.g. "rodero-cycle/checkpoints/t_0.1200" (no extension)
RESTART_FROM = "rodero-cycle/checkpoints/t_0.5060"

MMHG_S_PER_ML = 133.322 / 1e-6   # Pa s / m^3
ML_PER_MMHG = 1e-6 / 133.322     # m^3 / Pa

lv_params = CycleParams(
    t_zero=0.05, preload_pressure=500.0,
    t_end_diastole=0.12, p_end_diastole=1000.0,
    p_fill=100.0, period=0.8,
    windkessel=WindkesselParams(p_init=9000.0,
                                resistance=1.0 * MMHG_S_PER_ML,
                                compliance=1.5 * ML_PER_MMHG),
)
rv_params = CycleParams(
    t_zero=0.05, preload_pressure=170.0,
    t_end_diastole=0.12, p_end_diastole=330.0,
    p_fill=33.0, period=0.8,
    windkessel=WindkesselParams(p_init=3000.0,
                                resistance=0.1 * MMHG_S_PER_ML,
                                compliance=4.0 * ML_PER_MMHG),
)
PERIOD = lv_params.period

OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTDIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ geometry
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=GEODIR)
for name, tag in TAGS.items():
    assert int(geo.markers[name][0]) == tag, f"{name}: expected tag {tag}, file has {geo.markers[name]}"
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": QUAD_DEGREE})
mesh = geometry.mesh

# ------------------------------------------------------------------ model
material = pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **pulse.HolzapfelOgden.orthotropic_parameters())  # type: ignore
Ta = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")
model = pulse.CardiacModel(
    material=material,
    active=pulse.ActiveStress(geo.f0, activation=Ta),
    compressibility=pulse.compressibility.Compressible2(),
    viscoelasticity=pulse.viscoelasticity.Viscous(),
)

# ------------------------------------------------------------------ boundary conditions (no Neumann on cavities)
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
params["dt"] = pulse.Variable(DT, "s")
params["petsc_options"] = {
    **params["petsc_options"],
    "snes_monitor": None,
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "ksp_max_it": 100,
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-14,
    "ksp_converged_reason": None,
    "snes_lag_preconditioner": PRECONDITIONER_LAG,
    "snes_lag_preconditioner_persists": True,
}
problem = ControlledCavityDynamicProblem(model=model, geometry=geometry, bcs=bcs,
                                         cavities=cavities, parameters=params)
monitor = CavityMonitor(problem)

states = {"LV": CavityState(name="LV", params=lv_params),
          "RV": CavityState(name="RV", params=rv_params)}
controller = BiVCycleController(problem, monitor, states, restore_lag=PRECONDITIONER_LAG)

# ------------------------------------------------------------------ activation (periodic Bestel)
act_model = circulation.bestel.BestelActivation(parameters={
    "sigma_0": SIGMA_0, "t_sys": ACT_ONSET, "t_dias": ACT_ONSET + ACT_DURATION})
t_grid = np.linspace(0.0, PERIOD, int(round(PERIOD / 1e-4)) + 1)
act_grid = solve_ivp(act_model, [0.0, PERIOD], [0.0], t_eval=t_grid,
                     method="Radau", max_step=1e-3).y[0]


def activation(t):
    return float(np.interp(t % PERIOD, t_grid, act_grid))


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


# ------------------------------------------------------------------ checkpoint / restart
def write_checkpoint(t):
    base = CKPT_DIR / f"t_{t:.4f}"
    fields = (("u", problem.u), ("v", problem.v_old), ("a", problem.a_old))
    for i, (name, f) in enumerate(fields):
        mode = io4dolfinx.FileMode.write if i == 0 else io4dolfinx.FileMode.append
        io4dolfinx.write_function_on_input_mesh(f"{base}.bp", f, time=0.0, name=name,
                                                mode=mode, backend="adios2")
    pressures = {m: monitor.pressure(m) for m in states}
    if comm.rank == 0:
        data = {
            "t": t, "dt": DT, "pressures": pressures,
            "states": {m: {k: v for k, v in dataclasses.asdict(s).items() if k not in ("name", "params")}
                       for m, s in states.items()},
        }
        Path(f"{base}.json").write_text(json.dumps(data, indent=2))
        logger.info(f"Checkpoint written: {base}")


if RESTART_FROM is None:
    problem.solve()
    ref_volumes = {m: monitor.volume(m) for m in states}
    controller.initialize(0.0, ref_volumes)
    t_start = 0.0
else:
    base = str(RESTART_FROM)
    data = json.loads(Path(f"{base}.json").read_text())
    for name, f in (("u", problem.u), ("v", problem.v_old), ("a", problem.a_old)):
        io4dolfinx.read_function(f"{base}.bp", f, time=0.0, name=name, backend="adios2")
        f.x.scatter_forward()
    problem.u_old.x.array[:] = problem.u.x.array
    for m, s in states.items():
        for k, v in data["states"][m].items():
            setattr(s, k, v)
        monitor.init_pressure(m, data["pressures"][m])
    t_start = data["t"]
    if comm.rank == 0 and not np.isclose(data["dt"], DT):
        logger.warning(f"Checkpoint dt={data['dt']} differs from DT={DT}")

if comm.rank == 0:
    logger.info(f"Start t={t_start:.4f} s | LV V={states['LV'].volume_n * 1e6:.1f} mL, "
                f"RV V={states['RV'].volume_n * 1e6:.1f} mL")

# ------------------------------------------------------------------ time loop
vtx = dolfinx.io.VTXWriter(comm, OUTDIR / "displacement.bp", [problem.u], engine="BP4")
vtx.write(t_start)

header = ("t,Ta_Pa,phase_lv,V_lv_mL,P_lv_kPa,Part_lv_kPa,"
          "phase_rv,V_rv_mL,P_rv_kPa,Part_rv_kPa,newton_its,J_min,J_max,n_inverted")
log = []
n_total = int(round(T_END / DT))
k0 = int(round(t_start / DT))
ckpt_every = max(1, int(round(CHECKPOINT_EVERY / DT)))


def save_plots(arr):
    np.savetxt(OUTDIR / "log.csv", arr, delimiter=",", header=header, comments="")
    t = arr[:, 0]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].plot(t, arr[:, 4], label="LV")
    ax[0, 0].plot(t, arr[:, 8], label="RV")
    ax[0, 0].plot(t, arr[:, 5], "--", label="Art (sys)")
    ax[0, 0].plot(t, arr[:, 9], "--", label="Art (pul)")
    ax[0, 0].set_title("Pressure (kPa)")
    ax[0, 1].plot(t, arr[:, 3], label="LV")
    ax[0, 1].plot(t, arr[:, 7], label="RV")
    ax[0, 1].set_title("Volume (mL)")
    ax[0, 2].plot(arr[:, 3], arr[:, 4], label="LV")
    ax[0, 2].plot(arr[:, 7], arr[:, 8], label="RV")
    ax[0, 2].set_title("PV loop (kPa vs mL)")
    ax[1, 0].plot(t, arr[:, 1] / 1e3, "k", label="Ta (kPa)")
    axp = ax[1, 0].twinx()
    axp.step(t, arr[:, 2], where="post", label="LV phase")
    axp.step(t, arr[:, 6], where="post", label="RV phase")
    axp.set_yticks(list(PHASE_NAMES))
    axp.set_yticklabels(list(PHASE_NAMES.values()))
    ax[1, 0].set_title("Activation and phase")
    ax[1, 1].plot(t, arr[:, 11], label="J min")
    ax[1, 1].plot(t, arr[:, 12], label="J max")
    ax[1, 1].set_title("det F")
    ax[1, 2].plot(t, arr[:, 10], ".", label="its")
    ax[1, 2].set_title("Newton iterations")
    for a in ax.flat:
        a.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "summary.png")
    plt.close(fig)


for k in range(k0 + 1, n_total + 1):
    t = k * DT
    tai = activation(t)
    Ta.assign(tai)
    phases_before = {m: s.phase for m, s in states.items()}
    try:
        out = controller.step(t, DT)
    except Exception as ex:
        if comm.rank == 0:
            logger.error(f"Step failed at t={t:.4f} s "
                         f"(LV {PHASE_NAMES[phases_before['LV']]}, RV {PHASE_NAMES[phases_before['RV']]}): {ex}")
        break

    nit = problem.problem.solver.getIterationNumber()
    Jmin, Jmax, ninv = detF_stats()
    lv, rv = out["LV"], out["RV"]
    log.append([t, tai,
                lv["phase"], lv["V"] * 1e6, lv["P"] / 1e3, lv["P_art"] / 1e3,
                rv["phase"], rv["V"] * 1e6, rv["P"] / 1e3, rv["P_art"] / 1e3,
                nit, Jmin, Jmax, ninv])
    vtx.write(t)

    transition = any(out[m]["phase"] != phases_before[m] for m in states)
    if k % ckpt_every == 0 or transition:
        write_checkpoint(t)

    if comm.rank == 0:
        logger.info(
            f"t={t:.3f} Ta={tai / 1e3:6.1f} kPa | "
            f"LV {PHASE_NAMES[lv['phase']]:8s} V={lv['V'] * 1e6:6.1f} P={lv['P'] / 1e3:6.2f} Part={lv['P_art'] / 1e3:5.2f} | "
            f"RV {PHASE_NAMES[rv['phase']]:8s} V={rv['V'] * 1e6:6.1f} P={rv['P'] / 1e3:6.2f} Part={rv['P_art'] / 1e3:5.2f} | "
            f"its={nit} J=[{Jmin:.3f}, {Jmax:.3f}] inv={ninv}")
        if (k - k0) % PLOT_EVERY == 0:
            save_plots(np.array(log))

vtx.close()
if comm.rank == 0 and log:
    save_plots(np.array(log))
    logger.info(f"Finished at t={log[-1][0]:.4f} s; output in {OUTDIR}")