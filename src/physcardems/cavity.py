"""cavity_control.py

Per-cavity switchable constraint for pulse DynamicProblem.

Each cavity has one pressure unknown p, with the scalar equation
    volume mode   (mode = 1):  V(u) - V_target = 0
    pressure mode (mode = 0):  p - (A + B * V(u)) = 0
B = 0 gives a prescribed pressure (preload ramp, clamped filling pressure).
B != 0 couples pressure to volume inside Newton (Windkessel ejection, filling laws);
the coefficients are set by cycle_controller.apply_controls.
Do not also apply NeumannBCs on the cavity markers: the load comes from p.
"""
from dataclasses import dataclass, field

import dolfinx
import ufl
from mpi4py import MPI
import pulse
from petsc4py import PETSc

VOLUME_SCALE = 1e6     # m^3 -> mL for the volume residual
PRESSURE_SCALE = 1e-3  # Pa -> kPa for the pressure residual


class CavityControl:
    def __init__(self, mesh, name):
        def c(v):
            return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(v))
        self.name = name
        self.mode = c(0.0)       # 1 = volume, 0 = pressure
        self.V_target = c(0.0)   # m^3
        self.A = c(0.0)          # Pa
        self.B = c(0.0)          # Pa / m^3

    def set_volume(self, V):
        self.mode.value = 1.0
        self.V_target.value = V

    def set_pressure(self, P):
        self.mode.value = 0.0
        self.A.value = P
        self.B.value = 0.0

    def set_affine_pressure(self, A, B):
        self.mode.value = 0.0
        self.A.value = A
        self.B.value = B


@dataclass
class ControlledCavityDynamicProblem(pulse.problem.DynamicProblem):
    controls: dict = field(default_factory=dict)

    true_u_active: object = None  # active model evaluated at the true end-of-step displacement

    def _material_form(self, u, v, p):
        forms = super()._material_form(u, v, p)
        if self.true_u_active is not None:
            F = ufl.grad(self.u) + ufl.Identity(3)
            C = F.T * F
            var_C = ufl.grad(self.u_test).T * F + F.T * ufl.grad(self.u_test)
            forms[0] += ufl.inner(self.true_u_active.S(C, dev=True), 0.5 * var_C) * self.geometry.dx
        return forms

    def __post_init__(self):
        # constants must exist before the parent builds the forms
        self.controls = {cav.marker: CavityControl(self.geometry.mesh, cav.marker)
                         for cav in self.cavities}
        super().__post_init__()

    def _cavity_pressure_form(self, u, cavity_pressures=None):
        forms = self._empty_form()
        if self.num_cavity_pressure_states == 0:
            return forms
        V_u = self.geometry.volume_form(u)
        load = ufl.as_ufl(0.0)
        for i, cav in enumerate(self.cavities):
            ds = self.geometry.ds(self.geometry.markers[cav.marker][0])
            area = self.geometry.surface_area(cav.marker)
            p = cavity_pressures[i]
            q = self.cavity_pressures_test[i]
            ctl = self.controls[cav.marker]
            load += -p * V_u * ds  # same displacement-row term as pulse's cavity constraint
            r_vol = (ctl.V_target / area - V_u) * VOLUME_SCALE
            r_pres = ((ctl.A - p) / area + ctl.B * V_u) * PRESSURE_SCALE
            forms[1 + i] += (ctl.mode * r_vol + (1.0 - ctl.mode) * r_pres) * q * ds
        forms[0] += ufl.derivative(load, u, self.u_test)
        return forms


class CavityMonitor:
    """Precompiled current-volume forms and pressure readout, MPI-safe."""

    def __init__(self, problem):
        self.problem = problem
        g = problem.geometry
        self.index = {cav.marker: i for i, cav in enumerate(problem.cavities)}
        self.vol_forms = {
            m: dolfinx.fem.form(g.volume_form(problem.u) * g.ds(g.markers[m][0]))
            for m in self.index
        }

    def volume(self, marker):
        comm = self.problem.geometry.mesh.comm
        return comm.allreduce(dolfinx.fem.assemble_scalar(self.vol_forms[marker]), op=MPI.SUM)

    def pressure(self, marker):
        f = self.problem.cavity_pressures[self.index[marker]]
        return float(f.x.array[0])

    def init_pressure(self, marker, P):
        i = self.index[marker]
        self.problem.cavity_pressures[i].x.array[:] = P
        self.problem.cavity_pressures_old[i].x.array[:] = P


def set_preconditioner_lag(problem, lag):
    """Use lag=1 for the step after a phase switch, then restore (e.g. 20).
    petsc4py has no direct setter, so this goes through the options database."""
    snes = problem.problem.solver
    key = f"{snes.getOptionsPrefix() or ''}snes_lag_preconditioner"
    opts = PETSc.Options()
    opts[key] = lag
    snes.setFromOptions()
    del opts[key]