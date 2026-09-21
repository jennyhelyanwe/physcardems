"""zeta_active.py

ZetaSplitUFL with the mechanics time step held as a dolfinx Constant.

The parent builds dLambda and the zeta update from float(t - t_prev) when the UFL
is constructed. pulse compiles its forms once, at problem creation, when that
difference is zero, so inside Newton dLambda falls back to the stored previous
rate and Zetas reduces to Zetas_prev. With the step as a Constant, the compiled
form carries the actual rate dependence.
"""
import dolfinx
import ufl
from simcardemsx.backends import ZetaSplitUFL


class ZetaSplitConstDt(ZetaSplitUFL):
    def __init__(self, *, dt_ms, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.dt_const = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(dt_ms))

    @property
    def dt(self):
        return float(self.t.value - self._t_prev.value)

    def dLambda(self, lmbda):
        return (lmbda - self.lmbda_prev) / self.dt_const

    def _zeta(self, prev, A, c, lmbda):
        rate = A * self.dLambda(lmbda) - c * prev
        return ufl.max_value(prev + rate * (ufl.exp(-c * self.dt_const) - 1.0) / (-c), -1.0)

    def Zetas(self, lmbda):
        return self._zeta(self.Zetas_prev, self.As, self.cs, lmbda)

    def Zetaw(self, lmbda):
        return self._zeta(self.Zetaw_prev, self.Aw, self.cw, lmbda)