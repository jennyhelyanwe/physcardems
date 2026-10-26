# run_em_lvonly.py
# Coupled electromechanics of the left ventricle: electrophysiology (ToR-ORd cell model + monodomain),
# active tension (Land model), passive tissue mechanics and a five-phase cardiac cycle with a
# Windkessel afterload. The building blocks are in src/physcardems/lv_simulation.py; all settings
# come from one config file (see configs/lv_ellipsoid/baseline.toml).
#
# Usage: python3 scripts/run_em_lvonly.py configs/lv_ellipsoid/baseline.toml
# Outputs: runs/<case>/<config name>/ (summary.png, log.csv, metrics.json, simulation.bp, ...)
import argparse
import logging

from physcardems.lv_simulation import (
    load_settings, setup_geometry, setup_electrophysiology, setup_mechanics, setup_circulation,
    setup_outputs, initialise, time_steps, transfer_ep_to_mechanics, transfer_mechanics_to_ep,
)


def main(config_path):
    sim = load_settings(config_path)                  # config + case -> one settings object
    geo = setup_geometry(sim)                         # mesh, fibres, surfaces, cavity volume
    ep = setup_electrophysiology(sim, geo)            # cell model at every node, activation times, monodomain
    mech = setup_mechanics(sim, geo)                  # passive tissue + active tension, boundaries, cavity
    cycle = setup_circulation(sim, mech)              # five-phase cardiac cycle + Windkessel
    out = setup_outputs(sim, geo, ep, mech)           # log, figures, ParaView file, checkpoints
    t_start = initialise(sim, ep, mech, cycle, out)   # t = 0 (or a checkpoint, if [restart] is set)

    for t in time_steps(sim, t_start):                # one mechanics step (dt_mech_ms) at a time
        ep.advance_to(t)                              # EP substeps (dt_ep_ms) up to t
        transfer_ep_to_mechanics(ep, mech)            # crossbridge states -> active tension
        if not cycle.step(t):                         # solve mechanics with this phase's constraint
            break                                     # (solver failure: stop, keep the outputs so far)
        transfer_mechanics_to_ep(mech, ep)            # fibre stretch -> cell model
        out.record(t, ep, mech, cycle)                # log, figures, ParaView, checkpoints
    out.finish()                                      # final figures and metrics.json (EF, volumes, pressures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="LV coupled electromechanics simulation")
    ap.add_argument("config", help="simulation config, e.g. configs/lv_ellipsoid/baseline.toml")
    main(ap.parse_args().config)