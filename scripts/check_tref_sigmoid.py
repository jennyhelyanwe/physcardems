"""Apply a config's sigmoidal Tref to the stored isometric single-cell traces and report
resting and peak Ta per cell type, with and without the sigmoid. Writes tref_sigmoid_check.png.

    python3 scripts/check_tref_sigmoid.py configs/lv_ellipsoid/baseline.toml
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physcardems import parameters as mp
from physcardems.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("config")
args = ap.parse_args()
cfg, case = load_config(args.config)
land = mp.land_values(overrides=cfg["land"]["overrides"], scales=cfg["land"]["scales"])
phash = mp.parameter_hash(land, cfg["cell"]["ep_overrides"], cfg["cell"]["ep_scales"])
ssdir = case.root / f"steady_state_pcl{int(cfg['time']['pcl_ms'])}_{phash}"
ts = cfg["land"].get("tref_sigmoid")
assert ts, "config has no [land.tref_sigmoid] section"
tref_sys = land["Tref"]
tref_rest = tref_sys / cfg["land"]["scales"].get("Tref", 1.0) * ts["rest_scale"]


def tref_eff(n):
    return tref_rest + (tref_sys - tref_rest) / (1.0 + np.exp(-(n - ts["n_half"]) / ts["width"]))


fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
print(f"Tref {tref_rest:.0f} kPa at rest -> {tref_sys:.0f} kPa active; n_half {ts['n_half']}, width {ts['width']}")
for c, name in ((0, "endo"), (1, "epi"), (2, "mid")):
    meta = json.loads((ssdir / f"celltype{c}_{name}.json").read_text())
    z = np.load(ssdir / f"celltype{c}_{name}_last_beat.npz")
    n = z["Ta"] / meta["land"]["Tref"]              # isometric, lambda = 1 so h = 1
    ta_const = tref_sys * n
    ta_sig = tref_eff(n) * n
    print(f"{name:4s} n rest {n.min():.4f} peak {n.max():.3f} | constant Tref: rest {ta_const.min():6.2f} "
          f"peak {ta_const.max():6.1f} kPa | sigmoid: rest {ta_sig.min():5.2f} peak {ta_sig.max():6.1f} kPa")
    ax[c].plot(z["t"], ta_const, label=f"constant Tref {tref_sys:.0f} kPa")
    ax[c].plot(z["t"], ta_sig, label="sigmoidal Tref")
    ax[c].set(title=f"{name}, isometric", xlabel="time in beat (ms)", ylabel="Ta (kPa)", yscale="log")
    ax[c].legend()
fig.tight_layout()
fig.savefig("tref_sigmoid_check.png", dpi=110)
print("wrote tref_sigmoid_check.png")