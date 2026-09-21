# PhysCardEMS: Physiological Cardiac Electro-Mechanics Solver

`PhysCardEMS` is a FEniCSx-based framework for fully coupled cardiac electromechanics simulations, designed around physiological calibration and validation. It couples monodomain electrophysiology with active and passive myocardial mechanics and a five-phase cardiac cycle, and evaluates simulations against executable calibration and validation criteria derived from the ASME V&V40 framework ([Wang et al., eLife, 2025](https://elifesciences.org/reviewed-preprints/106555)).

The framework builds on [`simcardemsx`](https://github.com/ComputationalPhysiology/simcardemsx), [`fenicsx-pulse`](https://github.com/finsberg/fenicsx-pulse) and [`fenicsx-beat`](https://github.com/finsberg/fenicsx-beat).

## Features

- Coupled electromechanics: monodomain ToR-ORd electrophysiology with Land excitation-contraction coupling and Holzapfel-Ogden passive mechanics
- Cardiac cycle: five-phase cycle controller with a three-element Windkessel afterload model
- Pseudo-ECG computation (12-lead)
- Patient-specific geometry support, including spatially varying cell type and ionic scaling fields and Eikonal-driven activation
- Checkpoint and restart of mechanical and electrophysiological state
- Executable calibration and validation criteria, with reference ranges and sources defined in a single configuration file
- Population-based evaluation for sensitivity analysis: ranks Latin hypercube sampled populations against the criteria under user-defined importance profiles

## Status

`PhysCardEMS` is under active development. Calibration of the FEniCSx implementation against the criteria is ongoing, and interfaces may change between versions.

## Installation

TODO

## Getting started

TODO

## Calibration and validation criteria

Criteria are defined in `criteria.toml`. Each criterion specifies a biomarker, a reference range, units, a source, and whether it is used for calibration or validation. Simulations are scored by their normalised distance outside each reference range, so that near misses rank above large deviations. Importance profiles, either weighted or strictly tiered, allow populations to be ranked under different priorities.


## Automated tests

TODO: 

## Citing

If you use `PhysCardEMS` in your research, please cite:

https://elifesciences.org/reviewed-preprints/106555 This citation will be updated to the Version of Record once available. 

Please also cite the packages it builds on: `simcardemsx`, `fenicsx-pulse` and `fenicsx-beat` (see their repositories for citation details).

## Acknowledgements

The coupling approach builds on `simcardems` and `simcardemsx`, developed by Henrik Finsberg and colleagues at Simula Research Laboratory. 

Funding acknowledgements EPSRC Oxford IAA Partnership Fund. 

## Authors

- Zhinuo Jenny Wang (jenny.wang@citystgeorges.ac.uk)
