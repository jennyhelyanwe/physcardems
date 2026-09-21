# PhysCardEMS: Physiological Cardiac Electro-Mechanics Solver

PhysCardEMS is a FEniCSx-based framework for fully coupled cardiac electromechanics simulations, designed around physiological calibration and validation. It couples monodomain electrophysiology with active and passive myocardial mechanics and a five-phase cardiac cycle, and evaluates simulations against executable calibration and validation criteria derived from the ASME V&V40 framework ([Wang et al., eLife, 2025](https://elifesciences.org/reviewed-preprints/106555)).

The framework builds on [`simcardemsx`](https://github.com/ComputationalPhysiology/simcardemsx), [`fenicsx-pulse`](https://github.com/finsberg/fenicsx-pulse) and [`fenicsx-beat`](https://github.com/finsberg/fenicsx-beat).

## Features

- Coupled electromechanics: monodomain ToR-ORd electrophysiology with Land excitation-contraction coupling and Holzapfel-Ogden passive mechanics
- Cardiac cycle: five-phase cycle controller with a three-element Windkessel afterload model
- Pseudo-ECG computation (12-lead)
- Patient-specific geometry support, including spatially varying cell type and ionic scaling fields and Eikonal-driven activation
- Checkpoint and restart of mechanical and electrophysiological state
- Configuration-driven simulations, with the exact configuration and code version saved alongside each run
- Executable calibration and validation criteria, with reference ranges and sources defined in a single configuration file
- Population-based evaluation for sensitivity analysis: ranks sampled populations against the criteria under user-defined importance profiles

## Status

PhysCardEMS is under active development. Calibration of the FEniCSx implementation against the criteria is ongoing, and interfaces may change between versions. The coupled electromechanics script currently runs in serial.

## Installation

PhysCardEMS runs in the official FEniCSx Docker image (dolfinx v0.10.0). The tested package versions are recorded in `environment/`.

1. Clone the repository and start the container from its root:

```bash
   git clone https://github.com/jennyhelyanwe/physcardems.git
   cd physcardems
   bash docker/start_docker.sh
```

   The repository is mounted at `/home/shared` inside the container. Running the same script again re-attaches to the existing container.

2. Inside the container, install the dependencies and PhysCardEMS (once per container):

```bash
   bash docker/setup.sh
```

On HPC systems without Docker, the same image can be run with Apptainer or Singularity.

## Repository structure

```
physcardems/
├── src/physcardems/     library code (importable, no hard-coded paths)
│   └── data/            cell model (.ode) and criteria (criteria.toml)
├── scripts/             run scripts
├── cases/               geometries and their data, one folder per heart
├── configs/             simulation configurations, organised by study
├── runs/                simulation outputs (not tracked)
├── docker/              container start and setup scripts
├── environment/         tested package versions
└── dev/                 earlier development scripts, kept for reference
```

A simulation combines three things, each defined in its own place:

- **The heart**, in `cases/<name>/`. Each case folder holds the geometry and its data files, with a `case.toml` that says where the files are and how to read them: the geometry folder, facet and cell tags, the electrode file and its units. Only `case.toml` and conversion scripts are tracked in git; the data files are distributed separately (see below).
- **The simulation protocol**, in `configs/<study>/<name>.toml`. A config holds every setting for one simulation: time stepping, electrophysiology, passive and active mechanics, cell-model and Land parameters, circulation, and restart. Each config names the case it runs on. Units are given in the key names (for example `kappa_pa`, `t_end_ms`). Configs can also be written as JSON, which is convenient when generating them programmatically.
- **The model's base parameters**, in `src/physcardems/parameters.py`. This holds the literature base values for the ToR-ORd-Land parameters. Configs apply overrides and scalings on top of these, so each config states exactly how it departs from the base model.

### Case data

### Case data

The example case `cases/rodero_05/` includes the files needed to run it: the converted geometry, the electrode positions and the steady-state initial conditions for the default parameter set. The geometry is derived from the Rodero et al. virtual cohort (https://zenodo.org/records/4590294). Source files for regenerating the converted geometry are not included.
## Running a simulation

Inside the container, from `/home/shared`, pass a config to the run script:

```bash
python3 scripts/run_em.py configs/elife/em_tref7.toml
```

Output is written to `runs/<case>/<config name>/`, for example `runs/rodero_05/em_tref7/`. This includes:

- `config_used.json`, with the full configuration, the parameter hash and the git commit, so each result can be traced to the settings and code that produced it
- `log.csv`, with time series of pressures, volumes, phases, active tension, fibre stretch and solver diagnostics
- `pseudo_ecg.csv` and `pseudo_ecg.png`
- `summary.png`, with pressure, volume, PV loops, active tension and the 12-lead ECG
- `simulation.bp`, with all fields for visualisation in ParaView
- `checkpoints/`, for restarting (set `restart.from` in the config)

For a quick check that everything loads correctly, copy a config, set `t_end_ms` to a small value such as `20.0`, and run the copy.

To run a full beat in the background:

```bash
mkdir -p runs
nohup python3 scripts/run_em.py configs/elife/em_tref7.toml > runs/em_tref7.log 2>&1 &
```

### Steady-state initialisation

Each simulation starts from single-cell steady states for endocardial, epicardial and midmyocardial cells, stored in the case folder as `steady_state_pcl<cycle length>_<parameter hash>/`. The parameter hash identifies the cell-model parameters, and the run script stops if the steady state was paced with a different parameter set. To generate steady states, run the pacing script from inside the case folder:

```bash
cd cases/rodero_05
python3 ../../scripts/pace_steady_state.py --pcl 800
```

The pacing script currently takes its parameters from the defaults in `parameters.py`; converting it to read the same config as the run script is planned.

### Other scripts

`scripts/` also contains scripts for EP-only simulations (`run_ep.py`), mechanics-only cycles (`run_cycle.py`) and passive inflation sweeps (`passive_sweep.py`). These are being converted to the same config-driven structure as `run_em.py`, and some still take command-line arguments or in-script settings.

## Calibration and validation criteria

Criteria are defined in `src/physcardems/data/criteria.toml`. Each criterion specifies a biomarker, a reference range, units, a source, and whether it is used for calibration or validation. The current file holds healthy female reference ranges. Simulations are scored by their normalised distance outside each reference range, so that near misses rank above large deviations. Importance profiles, either weighted or strictly tiered, allow populations to be ranked under different priorities.

## Citing

If you use PhysCardEMS in your research, please cite:

Wang et al., eLife, https://elifesciences.org/reviewed-preprints/106555. This citation will be updated to the Version of Record once available.

Please also cite the packages it builds on: `simcardemsx`, `fenicsx-pulse` and `fenicsx-beat` (see their repositories for citation details).

## Acknowledgements

The coupling approach builds on `simcardems` and `simcardemsx`, developed by Henrik Finsberg and colleagues at Simula Research Laboratory. The ToR-ORd-Land cell model file (`src/physcardems/data/ToRORd_dynCl_endo_zetasplit.ode`) is taken from `simcardemsx`.

Funding: EPSRC Oxford IAA Partnership Fund.

## Authors

- Zhinuo Jenny Wang (jenny.wang@citystgeorges.ac.uk)