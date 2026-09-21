"""model_parameters.py

Single source for ToR-ORd-Land cell-model parameters, shared by the single-cell
pacing and the 3D run. Names are the gotranx .ode parameter names. A sensitivity
analysis changes values here, or passes its own dicts to the helper functions.

Values applied to the EP half go into the ODE parameter array; the subset the
Land tension uses on the mechanics side is passed to the ZetaSplit active model.
Parameters present in both halves (kws, kuw, rs, rw, phi) are set consistently.
"""
import hashlib
import json

from pathlib import Path
ODEFILE = Path(__file__).resolve().parent / "data" / "ToRORd_dynCl_endo_zetasplit.ode"

# Land parameters (Margara et al., PBMB), absolute values before scaling (.ode names)
LAND_BASE = {
    "Trpn50": 0.35,
    "ntrpn": 2.0,
    "ktrpn": 0.1,
    "rs": 0.25,
    "rw": 0.5,
    "Tot_A": 25.0,
    "ku": 0.021,         # Margara retune (simcardemsx .ode and legacy simcardems: 0.04)
    "ntm": 2.036,        # Margara retune (simcardemsx .ode and legacy simcardems: 2.4)
    "Beta0": 2.3,        # to be retuned against Moulin-like lambda traces
    "Beta1": -2.4,
    "gammas": 0.0085,
    "gammaw": 0.615,
    "phi": 2.23,
    "cat50_ref": 0.805,  # uM
    "Tref": 120.0,       # kPa
    "kws": 0.012,        # 1/ms
    "kuw": 0.182,        # 1/ms
}

# eLife calibration (Figure 1F), applied on top of LAND_BASE
LAND_OVERRIDES = {"cat50_ref": 0.534}
LAND_SCALES = {"kws": 3.86, "Tref": 3.0}

MECHANICS_ONLY_KEYS = ("Tref", "Beta0", "Tot_A")  # do not affect the paced EP states


# Other cell-model parameters (.ode names)
EP_OVERRIDES = {}
EP_SCALES = {
    "PCa_b": 2.0,        # TO CONFIRM: eLife two-fold GCaL (calibration step 3)
}

MECHANICS_KEYS = ("Beta0", "Tot_A", "Tref", "kuw", "kws", "phi", "rs", "rw")


def land_values(base=None, overrides=None, scales=None):
    land = dict(LAND_BASE if base is None else base)
    land.update(LAND_OVERRIDES if overrides is None else overrides)
    for k, s in (LAND_SCALES if scales is None else scales).items():
        land[k] = land[k] * s
    return land


def mechanics_parameters(land):
    """Dictionary for ZetaSplitUFL / ZetaSplitConstDt (replaces its whole parameter set)."""
    return {k: float(land[k]) for k in MECHANICS_KEYS}


def apply_to_ode_parameters(values, module, land, ep_overrides=None, ep_scales=None):
    """Write Land and cell-model parameters into a gotranx parameter array.

    values: array from module.init_parameter_values(), shape (num_params,) or (num_params, N).
    Returns the dict of names actually set, for logging.
    """
    applied = {}
    for k, v in land.items():
        if k in module.parameter:
            values[module.parameter_index(k)] = v
            applied[k] = float(v)
    for k, v in (EP_OVERRIDES if ep_overrides is None else ep_overrides).items():
        values[module.parameter_index(k)] = v
        applied[k] = float(v)
    for k, s in (EP_SCALES if ep_scales is None else ep_scales).items():
        idx = module.parameter_index(k)
        values[idx] = values[idx] * s
        applied[f"{k} (x{s})"] = float(s)
    return applied


def parameter_hash(land, ep_overrides=None, ep_scales=None, n=12):
    payload = {
        "land": {k: float(v) for k, v in sorted(land.items()) if k not in MECHANICS_ONLY_KEYS},
        "ep_overrides": dict(sorted((EP_OVERRIDES if ep_overrides is None else ep_overrides).items())),
        "ep_scales": dict(sorted((EP_SCALES if ep_scales is None else ep_scales).items())),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:n]