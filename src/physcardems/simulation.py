"""simulation.py: the building blocks of a biventricular coupled electromechanics simulation,
with simcardemsx (on fenicsx-beat and fenicsx-pulse) as the computational back end.

scripts/run_em.py strings these together:

    sim     = load_settings(config)                         what to simulate (config + case)
    cell    = load_cell_model(sim)                          the .ode, split into EP and contraction halves
    geo     = setup_geometry(sim)                           mesh, fibres, surfaces, reference scaling, masks
    ep      = setup_electrophysiology(sim, geo, cell)       cell model at every node + monodomain diffusion
    mech    = setup_mechanics(sim, geo, cell)               passive tissue + contraction + two cavities
    cycle   = setup_circulation(sim, mech)                  five-phase cycle per ventricle (pulse.cycle)
    coupled = setup_coupling(sim, ep, mech, cycle, cell)    simcardemsx's coupled time step + checkpoints
    out     = setup_outputs(sim, ep, mech, cycle, coupled)  log, figures, ParaView file, ECG

physcardems owns the case, the parameters and their steady states, the circulation settings and the
outputs. simcardemsx owns the coupled time step: the EP micro-steps, the transfers between EP and
mechanics, the contraction model stepped inside the mechanics Newton solve (monolithic), and the
rollback of a step that fails.

Units: the mesh is in metres; the config uses ms, mL, kPa and mmHg where marked; simcardemsx works
in ms, pulse and pulse.cycle in SI.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import time as timer
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import io4dolfinx
import beat
import pulse
from pulse.cycle import CycleController, CycleParams, PrescribedInflow, Windkessel
import cardiac_geometries.geometry

from simcardemsx.averaging import make_averager
from simcardemsx.backends import GeneratedActivation
from simcardemsx.checkpoint import Checkpointer
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Cycle
from simcardemsx.ode_model import load_ode_modules

from physcardems import parameters as mp
from physcardems import ecg as pseudo_ecg
from physcardems.config import load_config, save_config
from physcardems.pacing import ensure_steady_state
from physcardems.plotting import LOG_HEADER, PHASE_NAMES, ejection_metrics, load_ecg_csv, plot_summary

logger = logging.getLogger("em")

CHAMBERS = ("LV", "RV")
CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}
A_KEYS = ("a", "a_f", "a_s", "a_fs")      # Holzapfel-Ogden moduli, scaled in the valve plugs
MMHG_S_PER_ML = 133.322 / 1e-6            # mmHg s / mL -> Pa s / m^3
ML_PER_MMHG = 1e-6 / 133.322              # mL / mmHg -> m^3 / Pa
ML_PER_MS = 1e-6 / 1e-3                   # mL / ms -> m^3 / s

# log.csv: physcardems' 22 columns first, in their old positions (plot_summary, collect_sweep.py
# and plot_sweep.py read them by index), then the columns the simcardemsx back end adds.
# phase_lv / phase_rv are the phase after the step's switch (in force for the next step), as before;
# phase_solved_* are the phases the step was solved under.
EXTRA_COLUMNS = ("Ta_min_kPa", "phase_solved_lv", "phase_solved_rv", "snes_reason", "linear_its",
                 "solve_attempts", "first_newton_its", "first_snes_reason")
LOG_COLUMNS = tuple(LOG_HEADER.split(",")) + EXTRA_COLUMNS
ACTIVE_STATS_HEADER = ("t_ms,Ta_p5_kPa,Ta_p50_kPa,Ta_p95_kPa,"
                       "Ta_med_endo,lambda_med_endo,Ta_med_epi,lambda_med_epi,Ta_med_mid,lambda_med_mid")


class _Namespace(dict):
    """A dict with attribute access, for bundling related objects."""
    __getattr__ = dict.__getitem__


# ====================================================================== settings
@dataclasses.dataclass
class Settings:
    """Everything read from the config and case, in one place."""
    config_path: Path
    cfg: dict
    case: object
    outdir: Path
    # time
    dt_mech_ms: float
    dt_ep_ms: float
    t_end_ms: float
    pcl_ms: float
    output_every_ms: float
    checkpoint_every_ms: float
    plot_every: int
    restart: bool
    # cell model and Land
    land: dict
    ep_overrides: dict
    ep_scales: dict
    param_hash: str
    steady_state_dir: Path
    # geometry
    tags: dict
    myocardium_tags: tuple


def load_settings(config_path) -> Settings:
    cfg, case = load_config(config_path)
    t = cfg["time"]
    n_ep = int(round(t["dt_mech_ms"] / t["dt_ep_ms"]))
    assert np.isclose(n_ep * t["dt_ep_ms"], t["dt_mech_ms"]), "dt_mech_ms must be a multiple of dt_ep_ms"

    if cfg["land"].get("tref_sigmoid"):
        raise NotImplementedError(
            "[land.tref_sigmoid] is not available with the simcardemsx back end yet: the contraction "
            "model is generated from the .ode file, so a sigmoidal Tref has to be added there.")
    restart = cfg.get("restart", {})
    if restart.get("from"):
        raise ValueError("[restart] from = ... belongs to the old back end. To resume a run, set "
                         "[restart] continue = true: it continues from the checkpoint in the run's own folder.")

    land = mp.land_values(overrides=cfg["land"]["overrides"], scales=cfg["land"]["scales"])
    ep_overrides, ep_scales = cfg["cell"]["ep_overrides"], cfg["cell"]["ep_scales"]
    param_hash = mp.parameter_hash(land, ep_overrides, ep_scales)
    outdir = Path(cfg.get("outdir") or Path("runs") / case.name / Path(config_path).stem)
    return Settings(
        config_path=Path(config_path), cfg=cfg, case=case, outdir=outdir,
        dt_mech_ms=t["dt_mech_ms"], dt_ep_ms=t["dt_ep_ms"], t_end_ms=t["t_end_ms"], pcl_ms=t["pcl_ms"],
        output_every_ms=t["output_every_ms"], checkpoint_every_ms=t["checkpoint_every_ms"],
        plot_every=t["plot_every"], restart=bool(restart.get("continue", False)),
        land=land, ep_overrides=ep_overrides, ep_scales=ep_scales, param_hash=param_hash,
        steady_state_dir=case.root / f"steady_state_pcl{int(t['pcl_ms'])}_{param_hash}",
        tags=case.get("geometry", "tags"),
        myocardium_tags=tuple(case.get("geometry", "myocardium_cell_tags")),
    )


def physics(sim: Settings) -> dict:
    """What defines the run's physics, for the checkpoint: a restart must match it. Leaves out the
    run length, output settings and solver options, which a restart may change."""
    mechanics = {k: v for k, v in sim.cfg["mechanics"].items() if k != "preconditioner_lag"}
    return {
        "case": sim.case.name,
        "dt_mech_ms": sim.dt_mech_ms, "dt_ep_ms": sim.dt_ep_ms, "pcl_ms": sim.pcl_ms,
        "land": sim.land, "ep_overrides": sim.ep_overrides, "ep_scales": sim.ep_scales,
        "ep": sim.cfg["ep"], "mechanics": mechanics, "circulation": sim.cfg["circulation"],
    }


# ====================================================================== cell model
def load_cell_model(sim: Settings):
    """The .ode file split into its EP half (NumPy, for beat) and its contraction half (UFL, for the
    mechanics), generated once and cached in the run folder. Land parameters go to every half that
    declares them: the EP half through its parameter array (see Electrophysiology), the contraction
    half here."""
    modules = load_ode_modules(mp.ODEFILE, sim.outdir / "generated")
    mechanics_land = {k: float(v) for k, v in sim.land.items() if k in modules.mechanics.parameter}
    unused = sorted(set(sim.land) - set(mechanics_land) - set(modules.ep.parameter))
    if unused:
        raise KeyError(f"Land parameters {unused} are declared by neither half of the .ode split")
    logger.info(f"Land parameters for the contraction model: {mechanics_land}")
    return _Namespace(modules=modules, mechanics_land=mechanics_land)


# ====================================================================== geometry
def setup_geometry(sim: Settings):
    """Mesh, fibres and surface tags from the case folder; reference scaling; tissue masks."""
    geodir = sim.case.path("geometry", "dir")
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=MPI.COMM_WORLD, folder=geodir)
    for name, tag in sim.tags.items():
        assert int(geo.markers[name][0]) == tag, f"{name}: expected tag {tag}, file has {geo.markers[name]}"
    assert geo.cfun is not None, f"geometry in {geodir} has no cell tags"

    def cavity_volume_ml(marker):
        X, N = ufl.SpatialCoordinate(geo.mesh), ufl.FacetNormal(geo.mesh)
        ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(X, N) * ds(sim.tags[marker]))) * 1e6

    # uniform scaling of the imaged geometry about its centroid: a stand-in for unloading
    m = sim.cfg["mechanics"]
    assert (m.get("reference_scale") is None) != (m.get("reference_lv_volume_ml") is None), \
        "set exactly one of reference_scale or reference_lv_volume_ml"
    v_lv, v_rv = cavity_volume_ml("LV"), cavity_volume_ml("RV")
    ref_scale = m.get("reference_scale") or (m["reference_lv_volume_ml"] / v_lv) ** (1.0 / 3.0)
    xg = geo.mesh.geometry.x
    centroid = xg.mean(axis=0)
    xg[:] = centroid + ref_scale * (xg - centroid)
    logger.info(f"Reference scaling {ref_scale:.4f}: LV {v_lv:.1f} -> {cavity_volume_ml('LV'):.1f} mL, "
                f"RV {v_rv:.1f} -> {cavity_volume_ml('RV'):.1f} mL")

    geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": m["quad_degree"]})
    mesh = geometry.mesh

    # myocardium mask (no active tension in the valve plugs) and passive stiffness scale
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    myo_mask = dolfinx.fem.Function(DG0, name="myocardium")
    stiffness_scale = dolfinx.fem.Function(DG0, name="stiffness_scale")
    dofs = DG0.dofmap.list[geo.cfun.indices, 0]
    is_myo = np.isin(geo.cfun.values, sim.myocardium_tags)
    myo_mask.x.array[dofs] = is_myo.astype(float)
    stiffness_scale.x.array[dofs] = np.where(is_myo, 1.0, m["valve_stiffness_scale"])
    logger.info(f"Valve plug cells: {int(np.sum(~is_myo))} (Ta = 0, passive stiffness x{m['valve_stiffness_scale']})")

    return _Namespace(geo=geo, geometry=geometry, mesh=mesh, geodir=geodir, ref_scale=ref_scale,
                      myo_mask=myo_mask, stiffness_scale=stiffness_scale)


# ====================================================================== electrophysiology
class Electrophysiology:
    """ToR-ORd-Land cell model at every node (EP half of the zeta split) + monodomain diffusion.

    Every node is stimulated at its own activation time (lat_ms from the case), so the activation
    sequence is prescribed; diffusion smooths it and shapes repolarisation and the ECG. The arrays
    built here are handed to beat by reference: simcardemsx writes the contraction states and the
    fibre stretch back into them in place.
    """

    def __init__(self, sim: Settings, geo, cell):
        mesh = geo.mesh
        ep = cell.modules.ep
        self.sim, self.model = sim, ep
        self.W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        fields = {}
        for name in ("lat_ms", "celltype"):
            fields[name] = dolfinx.fem.Function(self.W, name=name)
            io4dolfinx.read_function(filename=geo.geodir / "ep_node_fields.bp", u=fields[name], name=name)
        self.lat = fields["lat_ms"].x.array.copy() - sim.cfg["ep"]["lat_shift_ms"]
        self.celltype = np.rint(fields["celltype"].x.array).astype(int)
        n_nodes = self.lat.size

        # initial states: single cells paced to steady state with these cell parameters
        ensure_steady_state(sim.steady_state_dir, sim.land, sim.ep_overrides, sim.ep_scales, sim.pcl_ms,
                            log=logger.info)
        self.steady_states = {c: json.loads((sim.steady_state_dir / f"celltype{c}_{n}.json").read_text())["states"]
                              for c, n in CELLTYPES.items()}
        _check_contraction_initial_states(self.steady_states, cell.modules.mechanics)
        names = sorted(ep.state, key=ep.state.get)
        missing_names = sorted(ep.missing, key=ep.missing.get)
        y0 = np.zeros((len(names), n_nodes))
        self.missing = np.zeros((len(missing_names), n_nodes))
        for c in CELLTYPES:
            ss = self.steady_states[c]
            y0[:, self.celltype == c] = np.array([ss[s] for s in names])[:, None]
            self.missing[:, self.celltype == c] = np.array([ss[s] for s in missing_names])[:, None]

        # parameters, one column per node: cell type, stimulus time, ionic and Land changes from the config
        self.P = np.repeat(ep.init_parameter_values(i_Stim_Period=sim.pcl_ms, lmbda=1.0, dLambda=0.0)[:, None],
                           n_nodes, axis=1)
        self.P[ep.parameter_index("celltype")] = self.celltype.astype(float)
        self.P[ep.parameter_index("i_Stim_Start")] = self.lat
        applied = mp.apply_to_ode_parameters(self.P, ep, sim.land, sim.ep_overrides, sim.ep_scales)
        logger.info(f"Parameter hash {sim.param_hash}; EP parameters applied: {applied}")
        self.iv, self.ica = ep.state_index("v"), ep.state_index("cai")

        # monodomain PDE + ODE splitting
        cond = beat.conductivities.default_conductivities(sim.cfg["ep"]["conductivities"])
        M = beat.conductivities.define_conductivity_tensor(f0=geo.geo.f0, **cond)
        C_m = (1.0 * beat.units.ureg("uF/cm**2")).to("uF/m**2").magnitude
        self.pde = beat.MonodomainModel(time=dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
                                        mesh=mesh, M=M, I_s=None, C_m=C_m)
        self.ode = beat.odesolver.DolfinODESolver(
            v_ode=dolfinx.fem.Function(self.W, name="v_ode"), v_pde=self.pde.state,
            fun=ep.generalized_rush_larsen, init_states=y0, parameters=self.P,
            num_states=y0.shape[0], v_index=self.iv,
            missing_variables=self.missing, num_missing_variables=self.missing.shape[0])
        self.solver = beat.MonodomainSplittingSolver(pde=self.pde, ode=self.ode)

        # wall time spent in EP, reset by the outputs after every mechanics step
        self.wall_s = 0.0
        step = self.solver.step

        def timed_step(*args, **kwargs):
            w0 = timer.perf_counter()
            try:
                return step(*args, **kwargs)
            finally:
                self.wall_s += timer.perf_counter() - w0

        self.solver.step = timed_step

        # pseudo-ECG at the case's electrode positions
        electrodes = pseudo_ecg.load_electrodes(sim.case.path("ecg", "electrodes"), sim.case.get("ecg", "unit_scale"))
        points, leads = pseudo_ecg.standard_leads(electrodes)
        self.ecg = pseudo_ecg.PseudoECG(self.pde.state, points, leads, sigma_b=1.0)
        self.ecg_every = max(1, int(round(sim.cfg["ep"]["dt_ecg_ms"] / sim.dt_ep_ms)))
        logger.info(f"EP: {n_nodes} nodes, {y0.shape[0]} states, activation {self.lat.min():.1f} to "
                    f"{self.lat.max():.1f} ms")

    def on_ep_step(self, t_ms, ep_step_idx):
        """Called by simcardemsx after every EP micro-step."""
        if ep_step_idx % self.ecg_every == 0:
            self.ecg.compute(t_ms)


def _check_contraction_initial_states(steady_states, mechanics_module):
    """The contraction model starts every quadrature point from the .ode's default states. That is
    right only if the paced steady states agree with them (they do for isometric pacing, where the
    distortion states stay at 0); stop otherwise."""
    names = sorted(mechanics_module.state, key=mechanics_module.state.get)
    defaults = dict(zip(names, map(float, mechanics_module.init_state_values())))
    differ = {(c, n): ss[n] for c, ss in steady_states.items() for n, d in defaults.items()
              if n in ss and not np.isclose(ss[n], d)}
    if differ:
        raise ValueError(f"The steady states' contraction states {differ} differ from the .ode defaults "
                         f"{defaults}, which the contraction model starts from")


def setup_electrophysiology(sim: Settings, geo, cell) -> Electrophysiology:
    return Electrophysiology(sim, geo, cell)


# ====================================================================== mechanics
class SolveLog:
    """Wraps problem.solve to keep the outcome of every solve (pulse.cycle retries a failed solve
    once, so a mechanics step is one or two solves)."""

    def __init__(self, problem):
        self.snes = problem.problem.solver
        self.attempts = []
        self._solve = problem.solve
        problem.solve = self

    def __call__(self, *args, **kwargs):
        converged = self._solve(*args, **kwargs)
        self.attempts.append({"iterations": int(self.snes.getIterationNumber()),
                              "reason": int(self.snes.getConvergedReason()),
                              "linear_iterations": int(self.snes.getLinearSolveIterations())})
        return converged


class Mechanics:
    """Passive Holzapfel-Ogden tissue (stiffer in the valve plugs) + the Land contraction model
    generated from the .ode, dynamic, with an epicardial spring and damper, and two cavities whose
    constraint (volume, pressure or Windkessel) the cycle switches phase by phase."""

    def __init__(self, sim: Settings, geo, cell):
        mesh = geo.mesh
        self.sim, self.geo, self.mesh = sim, geo, mesh
        m = sim.cfg["mechanics"]

        # contraction: the mechanics half of the .ode, stepped inside Newton (monolithic)
        self.backend = GeneratedActivation(
            cell.modules.mechanics, mesh, geo.geo.f0, quadrature_degree=m["quad_degree"],
            parameters=cell.mechanics_land, tension_scale=geo.myo_mask)
        logger.info(f"Active tension: Tref {cell.mechanics_land.get('Tref', float('nan')):.0f} kPa")

        # passive tissue: the four moduli are DG0 fields, x valve_stiffness_scale in the plugs
        def modulus(value):
            field = dolfinx.fem.Function(geo.stiffness_scale.function_space)
            field.x.array[:] = value * geo.stiffness_scale.x.array
            return pulse.Variable(field, "kPa")

        p = m["material"]
        ho = {k: modulus(p[k]) if k in A_KEYS else pulse.Variable(p[k], "dimensionless")
              for k in ("a", "b", "a_f", "b_f", "a_s", "b_s", "a_fs", "b_fs")}
        no_compression = not m["fibre_compression_resistance"]
        model = pulse.CardiacModel(
            material=pulse.HolzapfelOgden(f0=geo.geo.f0, s0=geo.geo.s0, use_heaviside=no_compression,
                                          use_subplus=no_compression, **ho),
            active=self.backend,
            compressibility=pulse.compressibility.Compressible2(kappa=pulse.Variable(m["kappa_pa"], "Pa")),
            viscoelasticity=pulse.viscoelasticity.Viscous(),
        )

        # epicardial spring and damper, nothing on the base
        const = lambda v: dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(v))
        b = m.get("boundary", {})
        bcs = pulse.BoundaryConditions(robin=(
            pulse.RobinBC(value=pulse.Variable(const(b.get("epi_spring_pa_per_m", 1e8)), "Pa / m"),
                          marker=sim.tags["EPI"]),
            pulse.RobinBC(value=pulse.Variable(const(b.get("epi_damping_pa_s_per_m", 5e3)), "Pa s/ m"),
                          marker=sim.tags["EPI"], damping=True),
        ))

        petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"] | {
            "snes_monitor": None, "ksp_type": "gmres", "ksp_gmres_restart": 100, "ksp_max_it": 100,
            "ksp_rtol": 1e-6, "ksp_atol": 1e-14, "ksp_converged_reason": None,
            "snes_lag_preconditioner": m["preconditioner_lag"], "snes_lag_preconditioner_persists": True}
        logger.info("Building mechanics problem (form compilation can take several minutes)")
        self.problem = pulse.DynamicProblem(
            model=model, geometry=geo.geometry, bcs=bcs,
            cavities=[pulse.problem.Cavity(marker=c, control=pulse.problem.CavityControl(mesh)) for c in CHAMBERS],
            parameters={"dt": pulse.Variable(sim.dt_mech_ms * 1e-3, "s"), "u_space": m.get("u_space", "P_2"),
                        "petsc_options": petsc_options},
        )
        self.solves = SolveLog(self.problem)
        logger.info("Mechanics problem ready")

        # diagnostics: myocardium at the contraction model's quadrature points, and det F per cell
        points = self.backend.space.element.interpolation_points
        myo_q = dolfinx.fem.Function(self.backend.space)
        myo_q.interpolate(dolfinx.fem.Expression(geo.myo_mask, points))
        self.in_myo = myo_q.x.array > 0.5
        X_ref = np.array([[0.25, 0.25, 0.25], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         dtype=dolfinx.default_real_type)
        self._J = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(self.problem.u)), X_ref)
        self._cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)

    def detF_stats(self):
        """min and max det F over each cell's vertices and centroid, and how many cells invert."""
        J = self._J.eval(self.mesh, self._cells)
        return float(J.min()), float(J.max()), int(np.sum(np.min(J, axis=1) <= 0.0))

    def active_tension_kpa(self):
        return self.backend.tension_kPa.x.array[self.in_myo]

    def fibre_stretch(self):
        return self.backend.outputs["lmbda"].x.array[self.in_myo]


def setup_mechanics(sim: Settings, geo, cell) -> Mechanics:
    return Mechanics(sim, geo, cell)


# ====================================================================== circulation
def cycle_params(c, pcl_ms) -> CycleParams:
    """[circulation.lv] or [circulation.rv] -> pulse.cycle.CycleParams, in SI."""
    mode = c.get("filling_mode", "volume")
    if mode != "volume":
        raise NotImplementedError(f"filling_mode = {mode!r}: pulse.cycle has prescribed-inflow filling only")
    wk = c["windkessel"]
    return CycleParams(
        t_zero=c["t_zero_s"], preload_pressure=c["preload_pressure_pa"],
        t_end_diastole=c["t_end_diastole_s"], p_end_diastole=c["p_end_diastole_pa"],
        p_fill=c["p_fill_pa"], period=pcl_ms * 1e-3,
        windkessel=Windkessel(
            p_init=wk["p_init_pa"],
            resistance=wk["resistance_mmhg_s_per_ml"] * MMHG_S_PER_ML,
            compliance=wk["compliance_ml_per_mmhg"] * ML_PER_MMHG,
            characteristic_impedance=wk["characteristic_impedance_mmhg_s_per_ml"] * MMHG_S_PER_ML),
        filling=PrescribedInflow(rate=c["filling_rate_ml_per_ms"] * ML_PER_MS),
    )


class Circulation:
    """Five-phase cardiac cycle for each ventricle: preload ramp, isovolumic contraction, ejection
    into a 3-element Windkessel, isovolumic relaxation, filling. Each phase sets the cavity's
    constraint. pulse.cycle retries a failed solve once, with a fresh preconditioner."""

    def __init__(self, sim: Settings, mech: Mechanics):
        circ = sim.cfg["circulation"]
        self.params = {"LV": cycle_params(circ["lv"], sim.pcl_ms), "RV": cycle_params(circ["rv"], sim.pcl_ms)}
        self.controller = CycleController(mech.problem, self.params,
                                          preconditioner_lag=sim.cfg["mechanics"]["preconditioner_lag"])

    def phases(self):
        """The phase each cavity is in now (the one the next step is solved under)."""
        return {c: int(self.controller.cycles[c].phase) for c in CHAMBERS}


def setup_circulation(sim: Settings, mech: Mechanics) -> Circulation:
    return Circulation(sim, mech)


# ====================================================================== the coupled step
class Coupled:
    """simcardemsx's SimulationController (EP micro-steps, transfers, mechanics solve, rollback of a
    failed step) and its Checkpointer (restart.bp and restart.json in the run folder)."""

    def __init__(self, sim: Settings, ep: Electrophysiology, mech: Mechanics, cycle: Circulation, cell):
        self.sim, self.ep, self.mech, self.cycle = sim, ep, mech, cycle
        self.controller = SimulationController(Cycle(cycle.controller), ep.solver, mech.backend, cell.modules,
                                               sim.dt_mech_ms, sim.dt_ep_ms)
        sim.outdir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = Checkpointer(self.controller, sim.outdir, physics=physics(sim))
        self.solved_under = cycle.phases()
        self.first_attempt = 0

    @property
    def t(self):
        return self.controller.t

    def initialise(self) -> float:
        """t = 0: one solve at zero cavity pressure, accepted, then both cavities start in PRELOAD.
        With [restart] continue = true: the state of the last checkpoint in the run folder."""
        if self.sim.restart:
            t0 = self.checkpointer.restore()
            logger.info(f"Restarted from the checkpoint in {self.sim.outdir} at t = {t0:.3f} ms")
            return t0
        self.controller.plan.forward(0.0)
        if not self.mech.problem.solve():
            raise RuntimeError("The unloaded solve did not converge")
        self.mech.backend.post_solve()
        self.controller.plan.backward()
        self.cycle.controller.initialize(0.0)
        V = {c: self.mech.geo.geometry.volume(c) * 1e6 for c in CHAMBERS}
        logger.info(f"Unloaded solve done; reference volumes LV {V['LV']:.1f} mL, RV {V['RV']:.1f} mL")
        return 0.0

    def step(self) -> bool:
        """One mechanics step (dt_mech_ms). Returns False if it failed; simcardemsx has then rolled
        every component back to the start of the step."""
        self.solved_under = self.cycle.phases()
        self.first_attempt = len(self.mech.solves.attempts)
        try:
            self.controller.step(ep_callback=self.ep.on_ep_step)
        except Exception as ex:
            names = {c: PHASE_NAMES[p] for c, p in self.solved_under.items()}
            logger.error(f"Coupled step from t = {self.t:.1f} ms failed (phases {names}): {ex}")
            return False
        return True

    def step_attempts(self):
        return self.mech.solves.attempts[self.first_attempt:]

    def phase_switched(self) -> bool:
        return self.cycle.phases() != self.solved_under


def setup_coupling(sim: Settings, ep, mech, cycle, cell) -> Coupled:
    return Coupled(sim, ep, mech, cycle, cell)


# ====================================================================== outputs
class Outputs:
    """log.csv + summary.png (every plot_every steps), a ParaView file, the pseudo-ECG, active-tension
    statistics, and checkpoints (every checkpoint_every_ms and at each phase switch)."""

    def __init__(self, sim: Settings, ep: Electrophysiology, mech: Mechanics, cycle: Circulation, coupled: Coupled):
        self.sim, self.ep, self.mech, self.cycle, self.coupled = sim, ep, mech, cycle, coupled
        sim.outdir.mkdir(parents=True, exist_ok=True)
        save_config(sim.cfg, sim.outdir / "config_used.json", param_hash=sim.param_hash, case=str(sim.case.root),
                    simcardemsx=importlib.metadata.version("simcardemsx"),
                    pulse=importlib.metadata.version("fenicsx-pulse"),
                    beat=importlib.metadata.version("fenicsx-beat"))
        mesh, W, backend = mech.mesh, ep.W, mech.backend

        # ParaView: everything on P1. Quadrature-point values (lambda, Zetas) are averaged onto P1,
        # as simcardemsx does for what it sends back to EP; Ta comes averaged from the backend.
        self.u_out = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))), name="u")
        self.v_out = dolfinx.fem.Function(W, name="v_mV")
        self.cai_out = dolfinx.fem.Function(W, name="cai_mM")
        self.ta_out = dolfinx.fem.Function(W, name="Ta_kPa")
        self.lam_out = dolfinx.fem.Function(W, name="lambda")
        self.zs_out = dolfinx.fem.Function(W, name="Zetas")
        self._average = [make_averager(backend.outputs["lmbda"], self.lam_out),
                         make_averager(backend.outputs["Zetas"], self.zs_out)]
        ct_out = dolfinx.fem.Function(W, name="celltype")
        ct_out.x.array[:] = ep.celltype
        lat_out = dolfinx.fem.Function(W, name="lat_ms")
        lat_out.x.array[:] = ep.lat
        myo_out = dolfinx.fem.Function(W, name="myocardium")
        make_averager(mech.geo.myo_mask, myo_out)()
        name = "simulation.bp" if not sim.restart else f"simulation_from_{coupled.t:07.1f}ms.bp"
        self.vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, sim.outdir / name,
                                        [self.u_out, self.v_out, self.cai_out, self.ta_out, self.lam_out,
                                         self.zs_out, ct_out, lat_out, myo_out], engine="BP4")

        # cell type at the quadrature points, for the active-tension statistics
        ct_q = dolfinx.fem.Function(backend.space)
        ct_q.interpolate(dolfinx.fem.Expression(ct_out, backend.space.element.interpolation_points))
        self.ct_q = np.rint(ct_q.x.array).astype(int)

        self.rows, self.active_rows = [], []
        if sim.restart:
            self._resume(coupled.t)
        self.t_wall = timer.time()

    def _resume(self, t0):
        """Keep the log and ECG rows up to the checkpoint, and the active-tension rows before it
        (write_fields adds the one at the checkpoint again)."""
        for path, rows, t_max in ((self.sim.outdir / "log.csv", self.rows, t0 + 1e-9),
                                  (self.sim.outdir / "active_stats.csv", self.active_rows, t0 - 1e-9)):
            if path.exists():
                old = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
                rows.extend(list(r) for r in old if r[0] <= t_max)
        te, ve, _ = load_ecg_csv(self.sim.outdir / "pseudo_ecg.csv")
        if te is not None:
            keep = te <= t0 + 1e-9
            self.ep.ecg.times[:] = list(te[keep])
            self.ep.ecg.values[:] = [list(v) for v in ve[keep]]

    # ---- fields for ParaView and active-tension statistics
    def write_fields(self, t_ms):
        mech, ep = self.mech, self.ep
        for average in self._average:
            average()
        self.ta_out.x.array[:] = mech.backend.active_tension.x.array
        self.u_out.interpolate(mech.problem.u)
        self.v_out.x.array[:] = ep.ode.values[ep.iv]
        self.cai_out.x.array[:] = ep.ode.values[ep.ica]
        self.vtx.write(t_ms)

        ta = mech.backend.tension_kPa.x.array
        lam = mech.backend.outputs["lmbda"].x.array
        myo = mech.in_myo
        row = [t_ms, *np.percentile(ta[myo], [5, 50, 95])]
        for c in CELLTYPES:
            sel = myo & (self.ct_q == c)
            row += [float(np.median(ta[sel])), float(np.median(lam[sel]))]
        self.active_rows.append(row)

    # ---- summary log + figure
    def save_summary(self):
        arr = np.array(self.rows)
        np.savetxt(self.sim.outdir / "log.csv", arr, delimiter=",", header=",".join(LOG_COLUMNS), comments="")
        ecg = self.ep.ecg
        plot_summary(arr, self.sim.outdir / "summary.png",
                     title=f"{self.sim.case.name} / {self.sim.config_path.stem}, parameter hash {self.sim.param_hash}, "
                           f"reference scale {self.mech.geo.ref_scale:.3f}",
                     ecg_t=ecg.times, ecg_v=ecg.values, ecg_leads=list(ecg.leads))
        ecg.save_csv(self.sim.outdir / "pseudo_ecg.csv")
        if self.active_rows:
            np.savetxt(self.sim.outdir / "active_stats.csv", np.array(self.active_rows), delimiter=",",
                       header=ACTIVE_STATS_HEADER, comments="")
        return arr

    # ---- called once per mechanics step
    def record(self):
        sim, mech, ep, coupled = self.sim, self.mech, self.ep, self.coupled
        t_ms = coupled.t
        rec = self.cycle.controller.records
        attempts = coupled.step_attempts()
        first, last = attempts[0], attempts[-1]
        Jmin, Jmax, ninv = mech.detF_stats()
        ta, lam, v = mech.active_tension_kpa(), mech.fibre_stretch(), ep.ode.values[ep.iv]
        wall = timer.time() - self.t_wall
        self.t_wall = timer.time()
        lv, rv = rec["LV"], rec["RV"]
        self.rows.append([
            t_ms, ta.max(), lam.min(), lam.max(),
            int(lv.phase), lv.V * 1e6, lv.P * 1e-3, lv.P_c * 1e-3,
            int(rv.phase), rv.V * 1e6, rv.P * 1e-3, rv.P_c * 1e-3,
            last["iterations"], Jmin, Jmax, ninv, v.min(), v.max(), ep.wall_s, wall,
            lv.Q * 1e6, rv.Q * 1e6,
            ta.min(), coupled.solved_under["LV"], coupled.solved_under["RV"],
            last["reason"], last["linear_iterations"], len(attempts), first["iterations"], first["reason"],
        ])
        logger.info(
            f"t={t_ms:6.1f} ms | Ta [{ta.min():6.2f}, {ta.max():6.1f}] kPa, lambda [{lam.min():.3f}, {lam.max():.3f}] | "
            f"LV {PHASE_NAMES[int(lv.phase)]:8s} V={lv.V * 1e6:6.1f} P={lv.P * 1e-3:6.2f} Q={lv.Q * 1e6:6.1f} | "
            f"RV {PHASE_NAMES[int(rv.phase)]:8s} V={rv.V * 1e6:6.1f} P={rv.P * 1e-3:6.2f} Q={rv.Q * 1e6:6.1f} | "
            f"its={last['iterations']} solves={len(attempts)} J=[{Jmin:.3f}, {Jmax:.3f}] inv={ninv} | "
            f"EP {ep.wall_s:.1f} s, total {wall:.1f} s")
        ep.wall_s = 0.0

        k = coupled.controller.mech_step_idx
        every = lambda ms: max(1, int(round(ms / sim.dt_mech_ms)))
        if k % every(sim.output_every_ms) == 0:
            self.write_fields(t_ms)
        if k % every(sim.checkpoint_every_ms) == 0 or coupled.phase_switched():
            coupled.checkpointer.write()
        if k % sim.plot_every == 0:
            self.save_summary()

    def finish(self):
        self.vtx.close()
        self.coupled.checkpointer.write()
        if not self.rows:
            return
        arr = self.save_summary()
        self.ep.ecg.plot(self.sim.outdir / "pseudo_ecg.png")
        for name, iV, iph in (("LV", 5, 4), ("RV", 9, 8)):
            edv, esv, ef, complete = ejection_metrics(arr[:, iV], arr[:, iph])
            if not np.isnan(edv):
                logger.info(f"{name}: EF {ef:.1f}%{'' if complete else ' (still ejecting)'}, "
                            f"EDV {edv:.1f} mL, ESV {esv:.1f} mL, SV {edv - esv:.1f} mL")
        logger.info(f"Finished at t = {arr[-1, 0]:.1f} ms; output in {self.sim.outdir}")


def setup_outputs(sim: Settings, ep, mech, cycle, coupled) -> Outputs:
    return Outputs(sim, ep, mech, cycle, coupled)