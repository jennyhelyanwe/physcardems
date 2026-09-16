"""cycle_controller.py

Port of biv_cavity_cycle_controller.py onto the monolithic cavity constraint
in cavity_control.py. SI units: Pa, s, m^3.

Phase -> constraint:
  PRELOAD             pressure: linear ramps (t_zero, then to t_end_diastole)
  ISOVOL_CONTRACTION  volume:   V = end_dia_vol
  EJECTION            pressure: P = P_n - (V - V_n)/C - dt P_n/(R C)   (affine in V)
  ISOVOL_RELAXATION   volume:   V = end_sys_vol
  FILLING             pressure: fill-rate or filling_gain law (affine in V)
"""
import dataclasses
import logging

from cavity_control import set_preconditioner_lag

logger = logging.getLogger(__name__)


class Phase:
    PRELOAD = 0
    ISOVOL_CONTRACTION = 1
    EJECTION = 2
    ISOVOL_RELAXATION = 3
    FILLING = 4


PHASE_NAMES = {0: "PRELOAD", 1: "IVC", 2: "EJECTION", 3: "IVR", 4: "FILLING"}


@dataclasses.dataclass
class WindkesselParams:
    p_init: float        # Pa
    compliance: float    # m^3 / Pa
    resistance: float    # Pa s / m^3
    evolve: bool = True


@dataclasses.dataclass
class CycleParams:
    t_zero: float                 # s
    preload_pressure: float       # Pa
    t_end_diastole: float         # s
    p_end_diastole: float         # Pa
    p_fill: float                 # Pa
    period: float                 # s
    windkessel: WindkesselParams
    gain_relaxation: tuple = (0.0, 0.0)   # (Pa / m^3, Pa s / m^3); legacy values need converting
    filling_gain: bool = False
    min_phase_duration: float = 0.05      # s (Alya's 0.05)
    dvol_eps: float = 1e-10               # m^3 (0.1 mm^3); flow-reversal threshold ending ejection
    min_fill_rate: float = -5e14          # Pa / m^3 (legacy -500 kPa/mm^3)


@dataclasses.dataclass
class CavityState:
    name: str
    params: CycleParams
    phase: int = Phase.PRELOAD
    n_beats: int = 0
    phase_counter: int = 0
    last_phase_change: float = 0.0
    volume_n: float = 0.0
    volume_n_minus_1: float = 0.0
    dvol_n: float = 0.0
    pressure_n: float = 0.0
    pressure_n_minus_1: float = 0.0
    wdk_pressure_n: float = 0.0
    ini_vol: float = 0.0
    end_preload_vol: float = 0.0
    end_dia_vol: float = 0.0
    end_sys_vol: float = 0.0


def preload_pressure(state: CavityState, t: float) -> float:
    p = state.params
    if t <= p.t_zero:
        return p.preload_pressure * t / p.t_zero
    if p.t_end_diastole > p.t_zero:
        return ((p.p_end_diastole - p.preload_pressure)
                * (t - state.n_beats * p.period - p.t_zero)
                / (p.t_end_diastole - p.t_zero) + p.preload_pressure)
    return state.pressure_n


def apply_controls(state: CavityState, ctl, t: float, dt: float) -> None:
    """Set this cavity's constraint for the step ending at t, from start-of-step state."""
    p = state.params
    if state.phase == Phase.PRELOAD:
        ctl.set_pressure(preload_pressure(state, t))
    elif state.phase == Phase.ISOVOL_CONTRACTION:
        ctl.set_volume(state.end_dia_vol)
    elif state.phase == Phase.EJECTION:
        wk = p.windkessel
        A = (state.wdk_pressure_n + state.volume_n / wk.compliance
             - dt * state.wdk_pressure_n / (wk.resistance * wk.compliance))
        ctl.set_affine_pressure(A, -1.0 / wk.compliance)
    elif state.phase == Phase.ISOVOL_RELAXATION:
        ctl.set_volume(state.end_sys_vol)
    elif state.phase == Phase.FILLING:
        if p.filling_gain:
            g0, g1 = p.gain_relaxation
            if state.volume_n > state.end_preload_vol:
                ctl.set_pressure(state.pressure_n - g0 * (state.volume_n - state.end_preload_vol))
            else:
                # P = P_n - g1 (V - V_n)/dt - g0 (V_n - V_endpreload)
                A = (state.pressure_n + g1 * state.volume_n / dt
                     - g0 * (state.volume_n - state.end_preload_vol))
                ctl.set_affine_pressure(A, -g1 / dt)
        else:
            k = (p.preload_pressure - p.p_fill) / max(abs(state.ini_vol - state.end_sys_vol), 1e-15)
            k = max(k, p.min_fill_rate)
            ctl.set_affine_pressure(state.pressure_n - k * state.volume_n, k)
    else:
        raise ValueError(f"Unknown phase {state.phase}")


def commit_step(state: CavityState, V: float, P: float) -> None:
    state.dvol_n = V - state.volume_n
    state.volume_n_minus_1 = state.volume_n
    state.pressure_n_minus_1 = state.pressure_n
    state.volume_n = V
    state.pressure_n = P


def advance_phase(state: CavityState, t: float) -> None:
    p = state.params
    since = t - state.last_phase_change
    if (state.phase == Phase.PRELOAD
            and t >= state.n_beats * p.period + p.t_end_diastole and t >= p.t_zero):
        state.phase, state.last_phase_change = Phase.ISOVOL_CONTRACTION, t
    elif state.phase == Phase.ISOVOL_CONTRACTION and state.pressure_n > state.wdk_pressure_n:
        state.phase, state.last_phase_change = Phase.EJECTION, t
    elif (state.phase == Phase.EJECTION and state.dvol_n > p.dvol_eps
          and since > p.min_phase_duration and state.volume_n < 0.99 * state.end_dia_vol):
        state.phase, state.last_phase_change = Phase.ISOVOL_RELAXATION, t
    elif state.phase == Phase.ISOVOL_RELAXATION and state.pressure_n < p.p_fill:
        state.phase, state.last_phase_change, state.phase_counter = Phase.FILLING, t, 0
    elif state.phase == Phase.FILLING:
        if p.period > 0.0 and t >= (state.n_beats + 1) * p.period + p.t_zero:
            state.phase, state.last_phase_change = Phase.PRELOAD, t
            state.n_beats += 1
        elif since > p.min_phase_duration:
            state.phase_counter = state.phase_counter + 1 if state.dvol_n < -1e-12 else 0
            if state.phase_counter > 5:
                state.last_phase_change = t
                state.n_beats += 1
                state.phase_counter = 0


class BiVCycleController:
    def __init__(self, problem, monitor, states: dict, restore_lag: int = 20):
        self.problem = problem
        self.monitor = monitor
        self.states = states          # {"LV": CavityState, "RV": CavityState}
        self.restore_lag = restore_lag
        self._refactor = True

    def initialize(self, t0: float, reference_volumes: dict, phase: int = Phase.PRELOAD,
                   pressures: dict | None = None) -> None:
        """reference_volumes: unloaded volumes (ini_vol). For a restart, pass the phase and
        the cavity pressures at t0; current volumes are read from the mechanics state."""
        for m, s in self.states.items():
            V = self.monitor.volume(m)
            P = 0.0 if pressures is None else pressures[m]
            s.ini_vol = reference_volumes[m]
            s.volume_n = s.volume_n_minus_1 = V
            s.end_preload_vol = s.end_dia_vol = V
            s.pressure_n = s.pressure_n_minus_1 = P
            s.wdk_pressure_n = s.params.windkessel.p_init
            s.phase, s.last_phase_change = phase, t0
            self.monitor.init_pressure(m, P)

    def _snapshot(self):
        pb = self.problem
        funcs = [pb.u, pb.u_old, pb.v_old, pb.a_old, *pb.cavity_pressures, *pb.cavity_pressures_old]
        return [(f, f.x.array.copy()) for f in funcs]

    @staticmethod
    def _restore(snap):
        for f, arr in snap:
            f.x.array[:] = arr

    def _solve(self):
        if self._refactor:
            set_preconditioner_lag(self.problem, 1)
        self.problem.solve()
        if self._refactor:
            set_preconditioner_lag(self.problem, self.restore_lag)
            self._refactor = False

    def _solve_with_retry(self, snap):
        try:
            self._solve()
        except Exception as ex:
            logger.warning(f"Solve failed ({ex}); restoring step state and retrying with a fresh factorisation")
            self._restore(snap)
            self._refactor = True
            self._solve()

    def step(self, t: float, dt: float) -> dict:
        snap = self._snapshot()
        for m, s in self.states.items():
            apply_controls(s, self.problem.controls[m], t, dt)
        self._solve_with_retry(snap)

        # legacy clamp: filling pressure not below the preload pressure
        clamp = [m for m, s in self.states.items()
                 if s.phase == Phase.FILLING and not s.params.filling_gain
                 and self.monitor.pressure(m) < s.params.preload_pressure]
        if clamp:
            self._restore(snap)
            for m in clamp:
                self.problem.controls[m].set_pressure(self.states[m].params.preload_pressure)
            self._refactor = True
            self._solve_with_retry(snap)

        changed = False
        out = {}
        for m, s in self.states.items():
            V, P = self.monitor.volume(m), self.monitor.pressure(m)
            if s.phase == Phase.PRELOAD:
                s.end_dia_vol = V
                if t <= s.params.t_zero:
                    s.end_preload_vol = V
            if s.phase == Phase.EJECTION:
                s.wdk_pressure_n = P       # cavity pressure equals arterial pressure while ejecting
            elif not s.params.windkessel.evolve:
                s.wdk_pressure_n = s.params.windkessel.p_init
            commit_step(s, V, P)
            old = s.phase
            advance_phase(s, t)
            if s.phase != old:
                changed = True
                if s.phase == Phase.ISOVOL_CONTRACTION:
                    s.end_dia_vol = V
                elif s.phase == Phase.ISOVOL_RELAXATION:
                    s.end_sys_vol = V
                logger.info(f"{m}: {PHASE_NAMES[old]} -> {PHASE_NAMES[s.phase]} at t={t:.4f} s, "
                            f"V={V * 1e6:.1f} mL, P={P / 1e3:.3f} kPa")
            out[m] = dict(phase=s.phase, V=V, P=P, P_art=s.wdk_pressure_n)
        self._refactor = self._refactor or changed
        return out