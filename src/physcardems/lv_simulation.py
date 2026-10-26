"""lv_simulation.py: the building blocks of an LV-only coupled electromechanics simulation.

scripts/run_em_lvonly.py strings these together:

    sim   = load_settings(config)                 what to simulate (config + case)
    geo   = setup_geometry(sim)                   mesh, fibres, surfaces, cavity volume
    ep    = setup_electrophysiology(sim, geo)     cell model at every node + monodomain diffusion
    mech  = setup_mechanics(sim, geo)             passive tissue + Land active tension + boundaries
    cycle = setup_circulation(sim, mech)          five-phase cardiac cycle with a Windkessel afterload
    out   = setup_outputs(sim, geo, ep, mech)     log, figures, ParaView file, checkpoints

and then, every mechanics time step: EP substeps -> crossbridge states to mechanics -> mechanics
solve under the current phase's constraint -> fibre stretch back to the cell model -> record.

Units: the mesh is in metres; the config uses ms, mL, kPa and mmHg where marked.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import time as timer
from pathlib import Path

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
from physcardems.pacing import ensure_steady_state
from physcardems.plotting_lv import log_from_rows, save_log, plot_summary, beat_biomarkers

logger = logging.getLogger("em")

MMHG_S_PER_ML = 133.322 / 1e-6      # mmHg s / mL -> Pa s / m^3
ML_PER_MMHG = 1e-6 / 133.322        # mL / mmHg -> m^3 / Pa
ML_PER_MS = 1e-6 / 1e-3             # mL / ms -> m^3 / s
A_KEYS = ("a", "a_f", "a_s", "a_fs")  # Holzapfel-Ogden stiffness values scaled by stiffness_scale
CELLTYPES = {0: "endo", 1: "epi", 2: "mid"}


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
    n_ep: int
    t_end_ms: float
    pcl_ms: float
    output_every_ms: float
    checkpoint_every_ms: float
    plot_every: int
    restart_from: str | None
    # cell model and Land
    land: dict
    ep_overrides: dict
    ep_scales: dict
    param_hash: str
    steady_state_dir: Path
    tref_sigmoid: dict | None
    # mechanics
    material: dict
    boundary: dict
    base_mode: str
    base_axis: int | None
    # geometry
    tags: dict
    myocardium_tags: tuple


def load_settings(config_path) -> Settings:
    cfg, case = load_config(config_path)
    assert list(case.get("geometry", "cavities", default=[])) == ["LV"], \
        "this simulation needs an LV-only case ([geometry] cavities = [\"LV\"]); use run_em.py for BiV"
    t, m = cfg["time"], cfg["mechanics"]
    n_ep = int(round(t["dt_mech_ms"] / t["dt_ep_ms"]))
    assert np.isclose(n_ep * t["dt_ep_ms"], t["dt_mech_ms"]), "dt_mech_ms must be a multiple of dt_ep_ms"

    land = mp.land_values(overrides=cfg["land"]["overrides"], scales=cfg["land"]["scales"])
    ep_overrides, ep_scales = cfg["cell"]["ep_overrides"], cfg["cell"]["ep_scales"]
    param_hash = mp.parameter_hash(land, ep_overrides, ep_scales)

    material = dict(m["material"])
    scale = float(material.pop("stiffness_scale", 1.0))
    for k in A_KEYS:
        material[k] *= scale

    boundary = m.get("boundary", {})
    base_mode = boundary.get("base", "fixed_normal")
    assert base_mode in ("fixed_normal", "free"), f"unknown [mechanics.boundary] base = {base_mode}"

    outdir = Path(cfg.get("outdir") or Path("runs") / case.name / Path(config_path).stem)
    return Settings(
        config_path=Path(config_path), cfg=cfg, case=case, outdir=outdir,
        dt_mech_ms=t["dt_mech_ms"], dt_ep_ms=t["dt_ep_ms"], n_ep=n_ep, t_end_ms=t["t_end_ms"],
        pcl_ms=t["pcl_ms"], output_every_ms=t["output_every_ms"],
        checkpoint_every_ms=t["checkpoint_every_ms"], plot_every=t["plot_every"],
        restart_from=cfg.get("restart", {}).get("from") or None,
        land=land, ep_overrides=ep_overrides, ep_scales=ep_scales, param_hash=param_hash,
        steady_state_dir=case.root / f"steady_state_pcl{int(t['pcl_ms'])}_{param_hash}",
        tref_sigmoid=cfg["land"].get("tref_sigmoid"),
        material=material, boundary=boundary, base_mode=base_mode,
        base_axis=case.get("geometry", "base_normal_axis"),
        tags=case.get("geometry", "tags"),
        myocardium_tags=tuple(case.get("geometry", "myocardium_cell_tags")),
    )


def time_steps(sim: Settings, t_start_ms: float):
    """Mechanics time points (ms) from just after t_start_ms to t_end_ms."""
    k0 = int(round(t_start_ms / sim.dt_mech_ms))
    for k in range(k0 + 1, int(round(sim.t_end_ms / sim.dt_mech_ms)) + 1):
        yield k * sim.dt_mech_ms


# ====================================================================== geometry
def setup_geometry(sim: Settings):
    """Mesh, fibres and surface tags from the case folder; reference scaling; volume origin."""
    comm = MPI.COMM_WORLD
    geodir = sim.case.path("geometry", "dir")
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    for name, tag in sim.tags.items():
        assert int(geo.markers[name][0]) == tag, f"{name}: expected tag {tag}, file has {geo.markers[name]}"

    def cavity_volume_ml():
        X, N = ufl.SpatialCoordinate(geo.mesh), ufl.FacetNormal(geo.mesh)
        ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(X, N) * ds(sim.tags["LV"]))) * 1e6

    # optional uniform scaling of the reference (unloaded) geometry
    m = sim.cfg["mechanics"]
    assert (m.get("reference_scale") is None) != (m.get("reference_lv_volume_ml") is None), \
        "set exactly one of reference_scale or reference_lv_volume_ml"
    v_ref = cavity_volume_ml()
    ref_scale = m.get("reference_scale") or (m["reference_lv_volume_ml"] / v_ref) ** (1.0 / 3.0)
    xg = geo.mesh.geometry.x
    centre = xg.mean(axis=0)
    if sim.base_axis is not None:
        centre[sim.base_axis] = 0.0          # scale about the basal plane so it stays at 0
    xg[:] = centre + ref_scale * (xg - centre)
    logger.info(f"Reference scaling {ref_scale:.4f}: LV {v_ref:.1f} -> {cavity_volume_ml():.1f} mL")

    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo, metadata={"quadrature_degree": m["quad_degree"]})
    mesh = geometry.mesh

    # Cavity volume = -1/3 integral of (x + u).n over the endocardium. That is exact while the open
    # base stays in the plane through the origin. With a free base, measure from the current basal
    # plane: volume_shift holds the mean basal displacement (updated each step, before any solve).
    volume_shift = dolfinx.fem.Constant(mesh, np.zeros(3, dtype=dolfinx.default_scalar_type))
    if sim.base_mode == "free":
        assert sim.base_axis is not None, "a free base needs [geometry] base_normal_axis in the case"
        X, N = ufl.SpatialCoordinate(mesh), ufl.FacetNormal(mesh)

        def volume_form(u=None):
            if u is None:
                return (-1.0 / 3.0) * ufl.dot(X - volume_shift, N)
            F = ufl.Identity(3) + ufl.grad(u)
            return (-1.0 / 3.0) * ufl.det(F) * ufl.dot(X + u - volume_shift, ufl.inv(F).T * N)

        geometry.volume_form = volume_form
    logger.info(f"Base: {sim.base_mode}")

    # myocardium mask (all cells in the ellipsoid; valve plugs in BiV cases are excluded from Ta)
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    myo_mask = dolfinx.fem.Function(DG0, name="myocardium")
    stiffness_field = dolfinx.fem.Function(DG0, name="stiffness_scale")
    dofs = DG0.dofmap.list[geo.cfun.indices, 0]
    is_myo = np.isin(geo.cfun.values, sim.myocardium_tags)
    myo_mask.x.array[dofs] = is_myo.astype(float)
    stiffness_field.x.array[dofs] = np.where(is_myo, 1.0, m["valve_stiffness_scale"])

    return _Namespace(geo=geo, geometry=geometry, mesh=mesh, geodir=geodir, ref_scale=ref_scale,
                      volume_shift=volume_shift, myo_mask=myo_mask, stiffness_field=stiffness_field)


class _Namespace(dict):
    """A dict with attribute access, for bundling related objects."""
    __getattr__ = dict.__getitem__


# ====================================================================== electrophysiology
class Electrophysiology:
    """ToR-ORd-Land cell model at every node (EP half of the zeta split) + monodomain diffusion.

    Every node is stimulated at its own activation time (lat_ms from the case), so the activation
    sequence is prescribed; diffusion smooths it and shapes repolarisation and the ECG.
    """

    def __init__(self, sim: Settings, geo):
        mesh = geo.mesh
        self.sim = sim
        self.W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        fields = {}
        for name in ("lat_ms", "celltype"):
            fields[name] = dolfinx.fem.Function(self.W, name=name)
            io4dolfinx.read_function(filename=geo.geodir / "ep_node_fields.bp", u=fields[name], name=name)
        self.lat = fields["lat_ms"].x.array.copy() - sim.cfg["ep"]["lat_shift_ms"]
        self.celltype = np.rint(fields["celltype"].x.array).astype(int)
        n_nodes = self.lat.size

        # EP half of the zeta-split ODE, generated once per run
        gen = sim.outdir / "generated" / "ep_zetasplit.py"
        gen.parent.mkdir(parents=True, exist_ok=True)
        if not gen.exists():
            ode_full = gotranx.load_ode(mp.ODEFILE)
            mech = ode_full.get_component("mechanics")
            code = gotranx.cli.gotran2py.get_code(
                ode_full - mech, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
                missing_values=mech.to_ode().missing_variables)
            gen.write_text(code)
        spec = importlib.util.spec_from_file_location("ep_zetasplit", gen)
        ep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ep)
        self.model = ep

        # initial states: single cells paced to steady state with these cell parameters
        ensure_steady_state(sim.steady_state_dir, sim.land, sim.ep_overrides, sim.ep_scales, sim.pcl_ms,
                            log=logger.info)
        ss = {c: json.loads((sim.steady_state_dir / f"celltype{c}_{n}.json").read_text())["states"]
              for c, n in CELLTYPES.items()}
        names = sorted(ep.state, key=ep.state.get)
        y0 = np.zeros((len(names), n_nodes))
        self.missing = np.zeros((len(ep.missing), n_nodes))
        for c in CELLTYPES:
            y0[:, self.celltype == c] = np.array([ss[c][s] for s in names])[:, None]
            for name, idx in ep.missing.items():
                self.missing[idx, self.celltype == c] = ss[c][name]

        # parameters: cell type, stimulus time, ionic and Land changes from the config
        self.P = np.repeat(ep.init_parameter_values(i_Stim_Period=sim.pcl_ms, lmbda=1.0, dLambda=0.0)[:, None],
                           n_nodes, axis=1)
        self.P[ep.parameter_index("celltype")] = self.celltype.astype(float)
        self.P[ep.parameter_index("i_Stim_Start")] = self.lat
        applied = mp.apply_to_ode_parameters(self.P, ep, sim.land, sim.ep_overrides, sim.ep_scales)
        logger.info(f"Parameter hash {sim.param_hash}; cell parameters applied: {applied}")
        self.i_lmbda = ep.parameter_index("lmbda")
        self.iv, self.ica, self.iXS, self.iXW = (ep.state_index(s) for s in ("v", "cai", "XS", "XW"))
        self.iZs, self.iZw = ep.missing_index("Zetas"), ep.missing_index("Zetaw")

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

        # pseudo-ECG at the case's electrode positions
        electrodes = pseudo_ecg.load_electrodes(sim.case.path("ecg", "electrodes"), sim.case.get("ecg", "unit_scale"))
        points, leads = pseudo_ecg.standard_leads(electrodes)
        self.ecg = pseudo_ecg.PseudoECG(self.pde.state, points, leads, sigma_b=1.0)
        self.ecg_every = max(1, int(round(sim.cfg["ep"]["dt_ecg_ms"] / sim.dt_ep_ms)))
        self.k_ep = 0
        self.wall_s = 0.0
        logger.info(f"EP: {n_nodes} nodes, {y0.shape[0]} states, activation {self.lat.min():.1f} to "
                    f"{self.lat.max():.1f} ms, {sim.n_ep} EP steps per mechanics step")

    def advance_to(self, t_ms):
        """EP substeps up to t_ms (dt_ep_ms each), recording the ECG."""
        w0 = timer.time()
        while self.k_ep * self.sim.dt_ep_ms < t_ms - 1e-9:
            t0 = self.k_ep * self.sim.dt_ep_ms
            self.k_ep += 1
            t1 = self.k_ep * self.sim.dt_ep_ms
            self.solver.step((t0, t1))
            if self.k_ep % self.ecg_every == 0:
                self.ecg.compute(t1)
        self.wall_s = timer.time() - w0


def setup_electrophysiology(sim: Settings, geo) -> Electrophysiology:
    return Electrophysiology(sim, geo)


# ====================================================================== mechanics
class Mechanics:
    """Passive Holzapfel-Ogden tissue + Land active tension, dynamic, with springs at the
    epicardium (and base) and one cavity (LV) whose pressure or volume the cycle controls."""

    def __init__(self, sim: Settings, geo):
        mesh = geo.mesh
        self.sim, self.geo, self.mesh = sim, geo, mesh
        m = sim.cfg["mechanics"]

        # active tension: Land model, zeta states in UFL (optionally with a sigmoidal Tref)
        land_mech = mp.mechanics_parameters(sim.land)
        kwargs = dict(f0=geo.geo.f0, s0=geo.geo.s0, n0=geo.geo.n0, mesh=mesh, dt_ms=sim.dt_mech_ms,
                      parameters=land_mech)
        if sim.tref_sigmoid:
            ts = sim.tref_sigmoid
            tref_rest = sim.land["Tref"] / sim.cfg["land"]["scales"].get("Tref", 1.0) * ts["rest_scale"]
            self.zeta = ZetaSplitSigmoidTref(tref_rest=tref_rest, n_half=ts["n_half"], width=ts["width"], **kwargs)
            logger.info(f"Active tension: Tref {tref_rest:.0f} kPa at rest -> {land_mech['Tref']:.0f} kPa when active")
        else:
            self.zeta = ZetaSplitConstDt(**kwargs)
            logger.info(f"Active tension: Tref {land_mech['Tref']:.0f} kPa")

        # passive tissue
        kpa = lambda v: pulse.Variable(v, "kPa")
        dim = lambda v: pulse.Variable(v, "dimensionless")
        p = sim.material
        ho = dict(a=kpa(p["a"]), b=dim(p["b"]), a_f=kpa(p["a_f"]), b_f=dim(p["b_f"]),
                  a_s=kpa(p["a_s"]), b_s=dim(p["b_s"]), a_fs=kpa(p["a_fs"]), b_fs=dim(p["b_fs"]))
        no_compression = not m["fibre_compression_resistance"]
        material = _Scaled(pulse.HolzapfelOgden(f0=geo.geo.f0, s0=geo.geo.s0, use_heaviside=no_compression,
                                                use_subplus=no_compression, **ho), geo.stiffness_field)
        model = pulse.CardiacModel(
            material=material,
            active=pulse.active_model.Passive(),   # active stress is added at the true displacement below
            compressibility=pulse.compressibility.Compressible2(kappa=pulse.Variable(m["kappa_pa"], "Pa")),
            viscoelasticity=pulse.viscoelasticity.Viscous(),
        )

        # boundary conditions
        const = lambda v: dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(v))
        b = sim.boundary
        robin = [pulse.RobinBC(value=pulse.Variable(const(b.get("epi_spring_pa_per_m", 1e8)), "Pa / m"),
                               marker=sim.tags["EPI"]),
                 pulse.RobinBC(value=pulse.Variable(const(b.get("epi_damping_pa_s_per_m", 5e3)), "Pa s/ m"),
                               marker=sim.tags["EPI"], damping=True)]
        if b.get("base_tangential_spring_pa_per_m", 0.0) > 0.0:
            robin.append(pulse.RobinBC(value=pulse.Variable(const(b["base_tangential_spring_pa_per_m"]), "Pa / m"),
                                       marker=sim.tags["BASE"], perpendicular=True))
        dirichlet = []
        if sim.base_axis is not None and sim.base_mode == "fixed_normal":
            base_facets = geo.geometry.facet_tags.find(sim.tags["BASE"])

            def base_normal_bc(V):
                Vi = V.sub(sim.base_axis)
                dofs = dolfinx.fem.locate_dofs_topological(Vi, mesh.topology.dim - 1, base_facets)
                return [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, Vi)]

            dirichlet.append(base_normal_bc)

        params = pulse.problem.DynamicProblem.default_parameters()
        params["dt"] = pulse.Variable(sim.dt_mech_ms / 1e3, "s")
        lag = m["preconditioner_lag"]
        params["petsc_options"] = {**params["petsc_options"], "snes_monitor": None, "ksp_type": "gmres",
                                   "ksp_gmres_restart": 100, "ksp_max_it": 100, "ksp_rtol": 1e-6,
                                   "ksp_atol": 1e-14, "ksp_converged_reason": None,
                                   "snes_lag_preconditioner": lag, "snes_lag_preconditioner_persists": True}
        logger.info("Building mechanics problem (form compilation can take several minutes)")
        self.problem = ControlledCavityDynamicProblem(
            model=model, geometry=geo.geometry,
            bcs=pulse.BoundaryConditions(robin=tuple(robin), dirichlet=tuple(dirichlet)),
            cavities=[pulse.problem.Cavity(marker="LV", volume=const(0.0))],
            parameters=params, true_u_active=_Scaled(self.zeta, geo.myo_mask))
        self.zeta.register(self.problem.u)
        self.monitor = CavityMonitor(self.problem)
        logger.info("Mechanics problem ready")

        # helpers for transfers, diagnostics and outputs
        self.DG1 = self.zeta.function_space
        self.myo_dg1 = dolfinx.fem.Function(self.DG1)
        self.myo_dg1.interpolate(geo.myo_mask)
        self.myo = self.myo_dg1.x.array > 0.5
        X_ref = np.array([[0.25, 0.25, 0.25], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         dtype=dolfinx.default_real_type)
        self._J = dolfinx.fem.Expression(ufl.det(ufl.Identity(3) + ufl.grad(self.problem.u)), X_ref)
        self._cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
        ds_base = ufl.Measure("ds", domain=mesh, subdomain_data=geo.geometry.facet_tags)(sim.tags["BASE"])
        self._base_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ds_base))
        ax = sim.base_axis if sim.base_axis is not None else 0
        self._ax = ax
        self._base_u = dolfinx.fem.form(self.problem.u[ax] * ds_base)
        self._base_u2 = dolfinx.fem.form(self.problem.u[ax] ** 2 * ds_base)

    def detF_stats(self):
        J = self._J.eval(self.mesh, self._cells)
        return float(J.min()), float(J.max()), int(np.sum(np.min(J, axis=1) <= 0.0))

    def base_motion(self):
        """Mean and standard deviation of the basal displacement along the long axis (m)."""
        mean = dolfinx.fem.assemble_scalar(self._base_u) / self._base_area
        var = max(dolfinx.fem.assemble_scalar(self._base_u2) / self._base_area - mean**2, 0.0)
        return mean, var ** 0.5

    def update_volume_origin(self):
        """Free base: measure the cavity volume from the current basal plane (lagged one step)."""
        self.base_dx, self.base_sd = self.base_motion()
        if self.sim.base_mode == "free":
            self.geo.volume_shift.value[self._ax] = self.base_dx

    def active_tension_kpa(self):
        return self.zeta.Ta_current.x.array[self.myo] / 1e3

    def fibre_stretch(self):
        return self.zeta.lmbda.x.array[self.myo]


class _Scaled:
    """Multiplies a pulse material or active model's stress and energy by a spatial field."""

    def __init__(self, model, scale):
        self._model, self._scale = model, scale

    def __getattr__(self, name):
        return getattr(self._model, name)

    def S(self, C, *args, **kwargs):
        return self._scale * self._model.S(C, *args, **kwargs)

    def P(self, F, *args, **kwargs):
        return self._scale * self._model.P(F, *args, **kwargs)

    def strain_energy(self, *args, **kwargs):
        return self._scale * self._model.strain_energy(*args, **kwargs)


def setup_mechanics(sim: Settings, geo) -> Mechanics:
    return Mechanics(sim, geo)


# ====================================================================== EP <-> mechanics
class _Transfer:
    """P1 (EP nodes) <-> DG1 (mechanics) work functions, created on first use."""
    fns = None


def _transfer_fns(ep, mech):
    if _Transfer.fns is None:
        _Transfer.fns = {k: dolfinx.fem.Function(ep.W) for k in ("XS", "XW", "Zs", "Zw", "lam")}
    return _Transfer.fns


def transfer_ep_to_mechanics(ep: Electrophysiology, mech: Mechanics):
    """Crossbridge states XS, XW from the cell model -> the Land tension in the mechanics."""
    f = _transfer_fns(ep, mech)
    f["XS"].x.array[:] = ep.ode.values[ep.iXS]
    f["XW"].x.array[:] = ep.ode.values[ep.iXW]
    mech.zeta.XS.interpolate(f["XS"])
    mech.zeta.XW.interpolate(f["XW"])


def transfer_mechanics_to_ep(mech: Mechanics, ep: Electrophysiology):
    """Distortion states Zetas, Zetaw and fibre stretch lambda -> the cell model."""
    f = _transfer_fns(ep, mech)
    f["Zs"].interpolate(mech.zeta._Zetas)
    f["Zw"].interpolate(mech.zeta._Zetaw)
    f["lam"].interpolate(mech.zeta.lmbda)
    ep.missing[ep.iZs, :] = f["Zs"].x.array
    ep.missing[ep.iZw, :] = f["Zw"].x.array
    ep.P[ep.i_lmbda, :] = f["lam"].x.array


def _init_zeta_from_ep(ep, mech):
    f = _transfer_fns(ep, mech)
    f["Zs"].x.array[:] = ep.missing[ep.iZs]
    f["Zw"].x.array[:] = ep.missing[ep.iZw]
    z = mech.zeta
    for target, src in ((z.Zetas_prev, f["Zs"]), (z._Zetas, f["Zs"]), (z.Zetaw_prev, f["Zw"]), (z._Zetaw, f["Zw"])):
        target.interpolate(src)


# ====================================================================== circulation
def cycle_params(c, pcl_ms):
    """[circulation.lv] -> CycleParams. filling_mode: "volume" (prescribed inflow, filling_rate_ml_per_ms),
    "pressure" (held at the filling pressure; refills as the tissue relaxes) or "gain" (Alya-style
    relaxation law, gain_relaxation_kpa_per_ml and gain_relaxation_kpa_ms_per_ml)."""
    wk = c["windkessel"]
    mode = c.get("filling_mode", "volume")
    assert mode in ("volume", "pressure", "gain"), f"unknown filling_mode {mode}"
    return CycleParams(
        t_zero=c["t_zero_s"], preload_pressure=c["preload_pressure_pa"],
        t_end_diastole=c["t_end_diastole_s"], p_end_diastole=c["p_end_diastole_pa"],
        p_fill=c["p_fill_pa"], period=pcl_ms / 1e3,
        windkessel=WindkesselParams(
            p_init=wk["p_init_pa"],
            resistance=wk["resistance_mmhg_s_per_ml"] * MMHG_S_PER_ML,
            compliance=wk["compliance_ml_per_mmhg"] * ML_PER_MMHG,
            characteristic_impedance=wk["characteristic_impedance_mmhg_s_per_ml"] * MMHG_S_PER_ML),
        filling_rate=c["filling_rate_ml_per_ms"] * ML_PER_MS if mode == "volume" else None,
        filling_gain=(mode == "gain"),
        gain_relaxation=((c.get("gain_relaxation_kpa_per_ml", 0.0) * 1e9,
                          c.get("gain_relaxation_kpa_ms_per_ml", 0.0) * 1e6) if mode == "gain" else (0.0, 0.0)),
    )


class Circulation:
    """Five-phase cardiac cycle for the LV: preload ramp, isovolumic contraction, ejection into a
    3-element Windkessel, isovolumic relaxation, filling. Each phase sets the cavity constraint."""

    def __init__(self, sim: Settings, mech: Mechanics):
        self.sim, self.mech = sim, mech
        self.states = {"LV": CavityState(name="LV", params=cycle_params(sim.cfg["circulation"]["lv"], sim.pcl_ms))}
        self.controller = BiVCycleController(mech.problem, mech.monitor, self.states,
                                             restore_lag=sim.cfg["mechanics"]["preconditioner_lag"])
        self.lv = None
        self.phase_changed = False

    def step(self, t_ms) -> bool:
        """Solve mechanics for the step ending at t_ms. Returns False if the solve failed."""
        mech = self.mech
        mech.zeta.t.value = t_ms
        mech.update_volume_origin()
        before = self.states["LV"].phase
        try:
            out = self.controller.step(t_ms / 1e3, self.sim.dt_mech_ms / 1e3)
        except Exception as ex:
            logger.error(f"Mechanics failed at t={t_ms:.1f} ms (LV {PHASE_NAMES[before]}): {ex}")
            return False
        mech.zeta.post_solve()
        self.lv = out["LV"]
        self.phase_changed = self.lv["phase"] != before
        return True


def setup_circulation(sim: Settings, mech: Mechanics) -> Circulation:
    return Circulation(sim, mech)


# ====================================================================== start / restart
def initialise(sim: Settings, ep: Electrophysiology, mech: Mechanics, cycle: Circulation, out) -> float:
    """Initial state at t = 0, or the state saved in a checkpoint. Writes the initial outputs and
    returns the start time (ms)."""
    t0 = _initial_state(sim, ep, mech, cycle)
    out.write_fields(t0)
    ep.ecg.compute(t0)
    return t0


def _initial_state(sim, ep, mech, cycle):
    _init_zeta_from_ep(ep, mech)
    if sim.restart_from is None:
        transfer_ep_to_mechanics(ep, mech)
        mech.zeta.t.value = 0.0
        mech.problem.solve()
        cycle.controller.initialize(0.0, {"LV": mech.monitor.volume("LV")})
        return 0.0

    base = str(sim.restart_from)
    data = json.loads(Path(f"{base}.json").read_text())
    assert np.isclose(data["dt_mech_ms"], sim.dt_mech_ms) and np.isclose(data["dt_ep_ms"], sim.dt_ep_ms)
    if data.get("parameter_hash") != sim.param_hash:
        logger.warning("Checkpoint parameter hash differs; valid only if the change cannot have acted yet")
    pb = mech.problem
    for name, f in (("u", pb.u), ("v", pb.v_old), ("a", pb.a_old)):
        io4dolfinx.read_function(f"{base}.bp", f, time=0.0, name=name, backend="adios2")
        f.x.scatter_forward()
    pb.u_old.x.array[:] = pb.u.x.array
    for m, s in cycle.states.items():
        for k, v in data["states"][m].items():
            setattr(s, k, v)
        mech.monitor.init_pressure(m, data["pressures"][m])
    saved = np.load(f"{base}_ep.npz")
    ep.ode.values[:] = saved["ode_values"]
    ep.missing[:] = saved["missing"]
    ep.P[ep.i_lmbda] = saved["lmbda_row"]
    for k, a in ZETA_FIELDS.items():
        getattr(mech.zeta, a).x.array[:] = saved[k]
    ep.ode.to_dolfin()
    ep.ode.ode_to_pde()
    ep.pde.assign_previous()
    t0 = data["t_ms"]
    mech.zeta.t.value = t0
    mech.zeta._t_prev.value = t0
    ep.k_ep = int(round(t0 / sim.dt_ep_ms))
    logger.info(f"Restarted from {base} at t={t0:.3f} ms")
    return t0


ZETA_FIELDS = {"XS": "XS", "XW": "XW", "Zetas": "_Zetas", "Zetaw": "_Zetaw", "Zetas_prev": "Zetas_prev",
               "Zetaw_prev": "Zetaw_prev", "lmbda": "lmbda", "lmbda_prev": "lmbda_prev",
               "dLambda": "_dLambda", "Ta": "Ta_current"}


# ====================================================================== outputs
class Outputs:
    """log.csv + summary.png (every plot_every steps), simulation.bp for ParaView, ECG, active-tension
    statistics, checkpoints (every checkpoint_every_ms and at each phase change), metrics.json at the end."""

    ACTIVE_FIELDS = ("lambda", "h_lambda", "active_fraction", "Zetas", "dLambda_per_ms", "Ta_kPa")

    def __init__(self, sim: Settings, geo, ep: Electrophysiology, mech: Mechanics):
        self.sim, self.ep, self.mech = sim, ep, mech
        sim.outdir.mkdir(parents=True, exist_ok=True)
        (sim.outdir / "checkpoints").mkdir(exist_ok=True)
        save_config(sim.cfg, sim.outdir / "config_used.json", param_hash=sim.param_hash, case=str(sim.case.root))
        mesh, W, DG1 = geo.mesh, ep.W, mech.DG1

        self.u_out = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))), name="u")
        self.v_out = dolfinx.fem.Function(W, name="v_mV")
        self.cai_out = dolfinx.fem.Function(W, name="cai_mM")
        ct_out = dolfinx.fem.Function(W, name="celltype")
        ct_out.x.array[:] = ep.celltype
        lat_out = dolfinx.fem.Function(W, name="lat_ms")
        lat_out.x.array[:] = ep.lat
        self.active_dg1 = {n: dolfinx.fem.Function(DG1, name=f"{n}_dg1") for n in self.ACTIVE_FIELDS}
        self.active_out = {n: dolfinx.fem.Function(W, name=n) for n in self.ACTIVE_FIELDS}
        w = ufl.TestFunction(W)
        dx = ufl.Measure("dx", domain=mesh)
        self.lumped = dolfinx.fem.assemble_vector(dolfinx.fem.form(w * dx)).array.copy()
        self.proj = {n: dolfinx.fem.form(self.active_dg1[n] * w * dx) for n in self.ACTIVE_FIELDS}
        self.vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, sim.outdir / "simulation.bp",
                                        [self.u_out, self.v_out, self.cai_out, *self.active_out.values(),
                                         ct_out, lat_out], engine="BP4")
        self.ct_dg1 = dolfinx.fem.Function(DG1)
        self.ct_dg1.interpolate(ct_out)
        self.beta0 = mech.zeta._parameters["Beta0"]
        self.rows, self.active_rows = [], []
        self.k = 0
        self.t_wall = timer.time()

    # ---- fields for ParaView and active-tension statistics
    def write_fields(self, t_ms):
        z, mech, ep = self.mech.zeta, self.mech, self.ep
        lam = z.lmbda.x.array
        lam_c = np.minimum(1.2, lam)
        h = np.maximum(0.0, 1.0 + self.beta0 * (lam_c + np.minimum(lam_c, 0.87) - 1.87))
        af = np.maximum(0.0, z.XS.x.array * (z._Zetas.x.array + 1.0) + z.XW.x.array * z._Zetaw.x.array)
        ta = z.Ta_current.x.array * mech.myo_dg1.x.array / 1e3
        for n, v in (("lambda", lam), ("h_lambda", h), ("active_fraction", af), ("Zetas", z._Zetas.x.array),
                     ("dLambda_per_ms", z._dLambda.x.array), ("Ta_kPa", ta)):
            self.active_dg1[n].x.array[:] = v
            self.active_out[n].x.array[:] = dolfinx.fem.assemble_vector(self.proj[n]).array / self.lumped
        self.u_out.interpolate(mech.problem.u)
        self.v_out.x.array[:] = ep.ode.values[ep.iv]
        self.cai_out.x.array[:] = ep.ode.values[ep.ica]
        self.vtx.write(t_ms)
        myo = mech.myo
        ct = np.rint(self.ct_dg1.x.array).astype(int)
        row = [t_ms, *np.percentile(ta[myo], [5, 50, 95]), float(np.mean(h[myo] == 0.0))]
        for c in (0, 1, 2):
            sel = myo & (ct == c)
            row += [float(np.median(ta[sel])), float(np.median(lam[sel]))]
        self.active_rows.append(row)

    # ---- checkpoint
    def write_checkpoint(self, t_ms, cycle):
        pb, ep, z = self.mech.problem, self.ep, self.mech.zeta
        base = self.sim.outdir / "checkpoints" / f"t_{t_ms:08.3f}"
        for i, (name, f) in enumerate((("u", pb.u), ("v", pb.v_old), ("a", pb.a_old))):
            mode = io4dolfinx.FileMode.write if i == 0 else io4dolfinx.FileMode.append
            io4dolfinx.write_function_on_input_mesh(f"{base}.bp", f, time=0.0, name=name, mode=mode, backend="adios2")
        np.savez(f"{base}_ep.npz", ode_values=ep.ode.values, missing=ep.missing, lmbda_row=ep.P[ep.i_lmbda],
                 **{k: getattr(z, a).x.array for k, a in ZETA_FIELDS.items()})
        data = {"t_ms": t_ms, "dt_mech_ms": self.sim.dt_mech_ms, "dt_ep_ms": self.sim.dt_ep_ms,
                "parameter_hash": self.sim.param_hash,
                "pressures": {m: self.mech.monitor.pressure(m) for m in cycle.states},
                "states": {m: {k: v for k, v in dataclasses.asdict(s).items() if k not in ("name", "params")}
                           for m, s in cycle.states.items()}}
        Path(f"{base}.json").write_text(json.dumps(data, indent=2))

    # ---- summary log + figure
    def save_summary(self):
        log = log_from_rows(self.rows)
        save_log(log, self.sim.outdir / "log.csv")
        plot_summary(log, self.sim.outdir / "summary.png",
                     title=f"{self.sim.case.name} / {self.sim.config_path.stem}, parameter hash {self.sim.param_hash}",
                     ecg_t=self.ep.ecg.times, ecg_v=self.ep.ecg.values, ecg_leads=list(self.ep.ecg.leads))
        self.ep.ecg.save_csv(self.sim.outdir / "pseudo_ecg.csv")
        if self.active_rows:
            np.savetxt(self.sim.outdir / "active_stats.csv", np.array(self.active_rows), delimiter=",",
                       comments="", header=("t_ms,Ta_p5_kPa,Ta_p50_kPa,Ta_p95_kPa,frac_h_zero,Ta_med_endo,"
                                            "lambda_med_endo,Ta_med_epi,lambda_med_epi,Ta_med_mid,lambda_med_mid"))
        return log

    # ---- called once per mechanics step
    def record(self, t_ms, ep, mech, cycle):
        sim, lv = self.sim, cycle.lv
        self.k += 1
        nit = mech.problem.problem.solver.getIterationNumber()
        Jmin, Jmax, ninv = mech.detF_stats()
        ta, lam, v = mech.active_tension_kpa(), mech.fibre_stretch(), ep.ode.values[ep.iv]
        wall = timer.time() - self.t_wall
        self.t_wall = timer.time()
        self.rows.append([t_ms, ta.max(), lam.min(), lam.max(), lv["phase"], lv["V"] * 1e6, lv["P"] / 1e3,
                          lv["P_art"] / 1e3, lv["Q"] * 1e6, nit, Jmin, Jmax, ninv, v.min(), v.max(), ep.wall_s, wall])
        logger.info(
            f"t={t_ms:6.1f} ms | Ta max {ta.max():6.1f} kPa, lambda [{lam.min():.3f}, {lam.max():.3f}] | "
            f"LV {PHASE_NAMES[lv['phase']]:8s} V={lv['V'] * 1e6:6.1f} mL P={lv['P'] / 1e3:6.2f} kPa "
            f"Q={lv['Q'] * 1e6:6.1f} mL/s | base dx {mech.base_dx * 1e3:+5.2f} mm (sd {mech.base_sd * 1e3:.2f}) | "
            f"its={nit} J=[{Jmin:.3f}, {Jmax:.3f}] inv={ninv} | EP {ep.wall_s:.1f} s, total {wall:.1f} s")
        every = lambda ms: max(1, int(round(ms / sim.dt_mech_ms)))
        if self.k % every(sim.output_every_ms) == 0:
            self.write_fields(t_ms)
        if self.k % every(sim.checkpoint_every_ms) == 0 or cycle.phase_changed:
            self.write_checkpoint(t_ms, cycle)
        if self.k % sim.plot_every == 0:
            self.save_summary()

    def finish(self):
        self.vtx.close()
        if not self.rows:
            return
        log = self.save_summary()
        self.ep.ecg.plot(self.sim.outdir / "pseudo_ecg.png")
        beats = beat_biomarkers(log)
        (self.sim.outdir / "metrics.json").write_text(json.dumps(
            {"t_reached_ms": float(log["t_ms"][-1]), "t_end_ms": self.sim.t_end_ms, "beats": beats}, indent=2))
        for b in beats:
            logger.info(f"beat {b['beat']}{'' if b['complete'] else ' (incomplete)'}: EF {b['LVEF_pct']:.1f}%, "
                        f"EDV {b['EDV_mL']:.1f} mL, ESV {b['ESV_mL']:.1f} mL, EDP {b['EDP_kPa']:.2f} kPa, "
                        f"Pmax {b['Pmax_kPa']:.1f} kPa")
        logger.info(f"Finished at t={log['t_ms'][-1]:.1f} ms; output in {self.sim.outdir}")


def setup_outputs(sim: Settings, geo, ep: Electrophysiology, mech: Mechanics) -> Outputs:
    return Outputs(sim, geo, ep, mech)