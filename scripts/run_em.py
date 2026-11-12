# run_em.py
# Coupled biventricular electromechanics: electrophysiology (ToR-ORd cell model + monodomain), active
# tension (Land model, generated from the same .ode), passive tissue mechanics with valve plugs, and a
# five-phase cardiac cycle with a Windkessel afterload for each ventricle. simcardemsx is the
# computational back end; the building blocks are in src/physcardems/simulation.py, and all settings
# come from one config file, which names its case. Serial only.
#
# Usage: python3 scripts/run_em.py configs/elife/em_tref7.toml
# Outputs: runs/<case>/<config name>/ (summary.png, log.csv, simulation.bp, restart.bp, ...)
# To continue a run from its last checkpoint, set [restart] continue = true in its config.
import argparse
import logging

from mpi4py import MPI

from physcardems.simulation import (
    load_settings, load_cell_model, setup_geometry, setup_electrophysiology, setup_mechanics,
    setup_circulation, setup_coupling, setup_outputs,
)


def main(config_path):
    assert MPI.COMM_WORLD.size == 1, "serial only for now"
    sim = load_settings(config_path)                       # config + case -> one settings object
    cell = load_cell_model(sim)                            # the .ode split into EP and contraction halves
    geo = setup_geometry(sim)                              # mesh, fibres, surfaces, reference scaling, masks
    ep = setup_electrophysiology(sim, geo, cell)           # cell model at every node, activation times, monodomain
    mech = setup_mechanics(sim, geo, cell)                 # passive tissue + contraction, boundaries, two cavities
    cycle = setup_circulation(sim, mech)                   # five-phase cardiac cycle + Windkessel, per ventricle
    coupled = setup_coupling(sim, ep, mech, cycle, cell)   # simcardemsx's coupled time step + checkpoints
    t_start = coupled.initialise()                         # unloaded solve at t = 0, or the last checkpoint
    out = setup_outputs(sim, ep, mech, cycle, coupled)     # log, figures, ParaView file, ECG
    out.write_fields(t_start)
    if not sim.restart:
        ep.ecg.compute(t_start)

    while coupled.t < sim.t_end_ms - 1e-9:                 # one mechanics step (dt_mech_ms) at a time:
        if not coupled.step():                             # EP substeps, states to mechanics, Newton solve
            break                                          # under this phase's constraint, stretch back to EP
        out.record()                                       # log, figures, ParaView, checkpoints
    out.finish()                                           # final figures, EF and volumes in the log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for lib in ("numba", "matplotlib", "scifem"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="Biventricular coupled electromechanics simulation")
    ap.add_argument("config", help="simulation config, e.g. configs/elife/em_tref7.toml")
    main(ap.parse_args().config)

