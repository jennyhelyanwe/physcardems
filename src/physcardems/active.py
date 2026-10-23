"""zeta_active.py

ZetaSplitUFL with the mechanics time step held as a dolfinx Constant.

The parent builds dLambda and the zeta update from float(t - t_prev) when the UFL
is constructed. pulse compiles its forms once, at problem creation, when that
difference is zero, so inside Newton dLambda falls back to the stored previous
rate and Zetas reduces to Zetas_prev. With the step as a Constant, the compiled
form carries the actual rate dependence.

ZetaSplitSigmoidTref additionally makes Tref a smooth function of activation, so a
large systolic Tref scaling does not scale the diastolic (resting) tension.
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


class ZetaSplitSigmoidTref(ZetaSplitConstDt):
    """Land tension with an activation-dependent Tref.

        n        = max(0, XS (Zetas + 1) + XW Zetaw) / rs     (normalised active fraction, 1 = full)
        Tref_eff = Tref_rest + (Tref - Tref_rest) / (1 + exp(-(n - n_half) / width))
        Ta       = h(lambda) Tref_eff n

    With Tref_rest at the unscaled Land value and Tref (from the parameters) at the scaled
    systolic value, resting tension stays near its unscaled level while the peak gets the
    full scaling. This is a phenomenological steepening of the tension-activation relation,
    not part of the published Land model. Tref values in kPa, as in parameters.py.
    """

    def __init__(self, *, tref_rest, n_half, width, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        c = lambda v: dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(v))
        self.tref_rest, self.n_half, self.width = c(tref_rest), c(n_half), c(width)

    def tref_effective(self, n):
        switch = 1.0 / (1.0 + ufl.exp(-(n - self.n_half) / self.width))
        return self.tref_rest + (self._parameters["Tref"] - self.tref_rest) * switch

    def Ta(self, lmbda):
        p = self._parameters
        lam_c = ufl.min_value(1.2, lmbda)
        h = ufl.max_value(0.0, 1.0 + p["Beta0"] * (lam_c + ufl.min_value(lam_c, 0.87) - 1.87))
        af = ufl.max_value(0.0, self.XS * (self.Zetas(lmbda) + 1.0) + self.XW * self.Zetaw(lmbda))
        n = af / p["rs"]
        return 1000 * h * self.tref_effective(n) * n   # Pa, as in ZetaSplitUFL.Ta