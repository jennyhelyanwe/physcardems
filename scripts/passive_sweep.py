"""passive_sweep.py

Quasi-static passive inflation of rodero_05 to end-diastole for a set of material
settings. Reports fibre stretch (lambda) and J over the myocardium (valve plugs
excluded), aiming for fibres that are uniformly stretched, not compressed.
Run serially: python3 passive_sweep.py [--cases name1,name2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import pulse
import cardiac_geometries.geometry

comm = MPI.COMM_WORLD
assert comm.size == 1, "run this serially"

GEODIR = Path("/home/shared/rodero_05/rodero_05_dolfinx_v2")
TAGS = {"LV": 30, "RV": 20, "EPI": 40, "BASE": 10}
MYOCARDIUM_TAGS = (1, 2)
QUAD_DEGREE = 4
DT = 2e-3                      # s, as in the coupled run
T_ZERO, T_ED = 0.05, 0.12      # s
P_LV_PRE, P_LV_ED = 500.0, 1000.0
P_RV_PRE, P_RV_ED = 170.0, 330.0
STATS_EVERY = 10               # steps between progress lines

def preload(t, p_pre, p_ed):
    if t <= T_ZERO:
        return p_pre * t / T_ZERO
    return p_pre + (p_ed - p_pre) * (t - T_ZERO) / (T_ED - T_ZERO)

H_ZERO_LAMBDA = (1.87 - 1.0 / 2.3) / 2.0   # h(lambda) = 0 below this (Beta0 = 2.3)

PULSE_DEFAULT = dict(a=0.059, b=8.023, a_f=18.472, b_f=16.026, a_s=2.481, b_s=11.120, a_fs=0.216, b_fs=11.436)
LEGACY = dict(a=0.61, a_f=1.56, b=7.5, b_f=35.31, a_s=0.7, b_s=33.24, a_fs=0.46, b_fs=5.09)

CASES = {
    "pulse_k1e6":        dict(params=PULSE_DEFAULT, kappa=1e6, compression=False, plug_scale=3.0),
    "pulse_comp_k1e6":   dict(params=PULSE_DEFAULT, kappa=1e6, compression=True,  plug_scale=3.0),
    "legacy_k1e6":       dict(params=LEGACY,        kappa=1e6, compression=False, plug_scale=3.0),
    "legacy_k1e7":       dict(params=LEGACY,        kappa=1e7, compression=False, plug_scale=3.0),
    "legacy_k1e8":       dict(params=LEGACY,        kappa=1e8, compression=False, plug_scale=3.0),
    "legacy_comp_k1e6":  dict(params=LEGACY,        kappa=1e6, compression=True,  plug_scale=3.0),
    "legacy_comp_k1e7":  dict(params=LEGACY,        kappa=1e7, compression=True,  plug_scale=3.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=",".join(CASES))
ap.add_argument("--outdir", default="passive_sweep")
args = ap.parse_args()
OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)


class ScaledModel:
    def __init__(self, model, scale):
        self._model = model
        self._scale = scale

    def __getattr__(self, name):
        return getattr(self._model, name)

    def S(self, C, *a, **k):
        return self._scale * self._model.S(C, *a, **k)

    def P(self, F, *a, **k):
        return self._scale * self._model.P(F, *a, **k)

    def strain_energy(self, *a, **k):
        return self._scale * self._model.strain_energy(*a, **k)


def ho_params(p):
    kpa = lambda v: pulse.Variable(v, "kPa")
    dim = lambda v: pulse.Variable(v, "dimensionless")
    return dict(a=kpa(p["a"]), b=dim(p["b"]), a_f=kpa(p["a_f"]), b_f=dim(p["b_f"]),
                a_s=kpa(p["a_s"]), b_s=dim(p["b_s"]), a_fs=kpa(p["a_fs"]), b_fs=dim(p["b_fs"]))


# ------------------------------------------------------------------ geometry (shared by all cases)
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=GEODIR)
for name, tag in TAGS.items():
    assert int(geo.markers[name][0]) == tag
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": QUAD_DEGREE})
mesh = geometry.mesh
tdim = mesh.topology.dim

DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
myo_mask = dolfinx.fem.Function(DG0, name="myocardium_dg0")
tag_dofs = DG0.dofmap.list[geo.cfun.indices, 0]
is_myo = np.isin(geo.cfun.values, MYOCARDIUM_TAGS)
myo_mask.x.array[tag_dofs] = is_myo.astype(float)
myo_cells = np.sort(geo.cfun.indices[is_myo]).astype(np.int32)

W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
Wv = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
w_test = ufl.TestFunction(W)
dxq = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": QUAD_DEGREE})
lumped = dolfinx.fem.assemble_vector(dolfinx.fem.form(w_test * dxq)).array.copy()
X_ref = np.array([[0.25, 0.25, 0.25], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dolfinx.default_real_type)


def pct(x, qs):
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def run_case(name, cfg):
    t0 = time.time()
    comp = cfg["compression"]
    scale = dolfinx.fem.Function(DG0)
    scale.x.array[tag_dofs] = np.where(is_myo, 1.0, cfg["plug_scale"])
    material = ScaledModel(
        pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **ho_params(cfg["params"]),
                             use_heaviside=not comp, use_subplus=not comp),
        scale,
    )
    model = pulse.CardiacModel(
        material=material,
        active=pulse.active_model.Passive(),
        compressibility=pulse.compressibility.Compressible2(kappa=pulse.Variable(cfg["kappa"], "Pa")),
        viscoelasticity=pulse.viscoelasticity.Viscous(),
    )
    p_lv = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")
    p_rv = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa")
    alpha_epi = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1e8)), "Pa / m")
    beta_epi = pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(5e3)), "Pa s/ m")
    bcs = pulse.BoundaryConditions(
        neumann=(pulse.NeumannBC(traction=p_lv, marker=TAGS["LV"]),
                 pulse.NeumannBC(traction=p_rv, marker=TAGS["RV"])),
        robin=(pulse.RobinBC(value=alpha_epi, marker=TAGS["EPI"]),
               pulse.RobinBC(value=beta_epi, marker=TAGS["EPI"], damping=True)),
    )
    params = pulse.problem.DynamicProblem.default_parameters()
    params["dt"] = pulse.Variable(DT, "s")
    params["petsc_options"] = {
        **params["petsc_options"],
        "ksp_type": "gmres", "ksp_gmres_restart": 100, "ksp_max_it": 100,
        "ksp_rtol": 1e-6, "ksp_atol": 1e-14,
        "snes_lag_preconditioner": 20, "snes_lag_preconditioner_persists": True,
    }
    problem = pulse.problem.DynamicProblem(model=model, geometry=geometry, bcs=bcs, parameters=params)

    F = ufl.Identity(3) + ufl.grad(problem.u)
    C = F.T * F
    lam_ufl = ufl.sqrt(ufl.inner(C * geo.f0, geo.f0))
    J_ufl = ufl.det(F)
    lam_expr = dolfinx.fem.Expression(lam_ufl, X_ref)
    J_expr = dolfinx.fem.Expression(J_ufl, X_ref)
    vol_lv = dolfinx.fem.form(geometry.volume_form(problem.u) * geometry.ds(TAGS["LV"]))
    vol_rv = dolfinx.fem.form(geometry.volume_form(problem.u) * geometry.ds(TAGS["RV"]))

    n_steps = int(round(T_ED / DT))
    total_its, steps_done, failed = 0, 0, False
    history = []
    for k in range(1, n_steps + 1):
        t = k * DT
        p_lv.assign(preload(t, P_LV_PRE, P_LV_ED))
        p_rv.assign(preload(t, P_RV_PRE, P_RV_ED))
        try:
            problem.solve()
        except Exception as ex:
            print(f"  [{name}] failed at t={t * 1e3:.0f} ms: {type(ex).__name__}: {ex}")
            failed = True
            break
        its = problem.problem.solver.getIterationNumber()
        total_its += its
        steps_done = k
        if k % STATS_EVERY == 0 or k == n_steps:
            lam = lam_expr.eval(mesh, myo_cells).ravel()
            history.append(dict(t_ms=t * 1e3, its=its, lam_p5=float(np.percentile(lam, 5)),
                                lam_p50=float(np.median(lam)), lam_p95=float(np.percentile(lam, 95))))
            print(f"  [{name}] t={t * 1e3:5.0f} ms, p_lv={preload(t, P_LV_PRE, P_LV_ED):6.0f} Pa: its={its}, "
                  f"lambda p5/p50/p95 = {history[-1]['lam_p5']:.3f}/{history[-1]['lam_p50']:.3f}/"
                  f"{history[-1]['lam_p95']:.3f}")
    frac = steps_done / n_steps
    n_steps = steps_done

    lam = lam_expr.eval(mesh, myo_cells).ravel()
    J = J_expr.eval(mesh, myo_cells).ravel()
    res = dict(
        case=name, failed=failed, load_reached=frac, load_steps=n_steps, newton_its=total_its,
        wall_s=time.time() - t0, kappa_Pa=cfg["kappa"], compression_resistance=comp,
        plug_scale=cfg["plug_scale"], params=cfg["params"],
        V_lv_mL=dolfinx.fem.assemble_scalar(vol_lv) * 1e6, V_rv_mL=dolfinx.fem.assemble_scalar(vol_rv) * 1e6,
        lambda_=pct(lam, (1, 5, 25, 50, 75, 95, 99)),
        lambda_spread_5_95=float(np.percentile(lam, 95) - np.percentile(lam, 5)),
        lambda_iqr=float(np.percentile(lam, 75) - np.percentile(lam, 25)),
        frac_lambda_lt_1=float(np.mean(lam < 1.0)),
        frac_lambda_lt_0p9=float(np.mean(lam < 0.9)),
        frac_h_zero=float(np.mean(lam < H_ZERO_LAMBDA)),
        frac_lambda_gt_1p2=float(np.mean(lam > 1.2)),
        J=pct(J, (1, 50, 99)),
        history=history,
    )

    u_p1 = dolfinx.fem.Function(Wv, name="u")
    u_p1.interpolate(problem.u)
    out = {}
    for fname, expr in (("lambda", lam_ufl), ("J", J_ufl), ("myocardium", myo_mask)):
        f = dolfinx.fem.Function(W, name=fname)
        f.x.array[:] = dolfinx.fem.assemble_vector(dolfinx.fem.form(expr * w_test * dxq)).array / lumped
        out[fname] = f
    with dolfinx.io.VTXWriter(comm, OUTDIR / f"{name}.bp", [u_p1, *out.values()], engine="BP4") as vtx:
        vtx.write(0.0)
    (OUTDIR / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res


results = []
for name in args.cases.split(","):
    name = name.strip()
    print(f"\n=== {name}: {CASES[name]}")
    results.append(run_case(name, CASES[name]))

cols = ["case", "load", "its", "V_lv", "V_rv", "lam_p5", "lam_p50", "lam_p95", "spread", "iqr",
        "<1", "<0.9", "h=0", ">1.2", "J_p1", "J_p99", "wall_s"]
lines = [",".join(cols)]
print("\n" + "  ".join(f"{c:>9s}" for c in cols))
for r in results:
    row = [r["case"], f"{r['load_reached']:.2f}", r["newton_its"], f"{r['V_lv_mL']:.1f}", f"{r['V_rv_mL']:.1f}",
           f"{r['lambda_']['p5']:.3f}", f"{r['lambda_']['p50']:.3f}", f"{r['lambda_']['p95']:.3f}",
           f"{r['lambda_spread_5_95']:.3f}", f"{r['lambda_iqr']:.3f}",
           f"{r['frac_lambda_lt_1']:.3f}", f"{r['frac_lambda_lt_0p9']:.3f}", f"{r['frac_h_zero']:.3f}",
           f"{r['frac_lambda_gt_1p2']:.3f}", f"{r['J']['p1']:.3f}", f"{r['J']['p99']:.3f}", f"{r['wall_s']:.0f}"]
    lines.append(",".join(str(x) for x in row))
    print("  ".join(f"{str(x):>9s}" for x in row))
(OUTDIR / "summary.csv").write_text("\n".join(lines) + "\n")
print(f"\nsummary in {OUTDIR / 'summary.csv'}")