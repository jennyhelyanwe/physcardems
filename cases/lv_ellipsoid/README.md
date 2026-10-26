# LV ellipsoid: coupled electromechanics of the left ventricle

This case simulates heartbeats of an idealised human left ventricle (LV), from the electrical
activity of individual heart cells to the pressure and volume of the whole chamber. It is a
teaching and research model: simple enough to run on a laptop, detailed enough to ask
physiological questions (what does a calcium channel blocker do to ejection fraction? how does
wall thickness change the pressure-volume loop?).

## What is in the model

| Part | What it does | Where it is set |
|---|---|---|
| Anatomy | Truncated ellipsoid, unloaded (zero-pressure) shape: 73 mL cavity, 9 mm wall, 96 g | `anatomy.toml` (this folder) |
| Fibres | Helix angle -60 to +60 degrees from endocardium to epicardium | `anatomy.toml` |
| Activation | Prescribed activation times from three early sites and a fast endocardial layer (QRS ~ 98 ms) | `anatomy.toml` |
| Cell electrophysiology | ToR-ORd human ventricular model at every node; endo, mid and epi cells | config `[cell]` |
| Active tension | Land model, driven by the cell's calcium and the local fibre stretch | config `[land]` |
| Passive tissue | Holzapfel-Ogden hyperelastic material | config `[mechanics.material]` |
| Boundaries | Spring on the epicardium (pericardium-like); free base | config `[mechanics.boundary]` |
| Circulation | Five phases: filling, isovolumic contraction, ejection into a 3-element Windkessel, isovolumic relaxation | config `[circulation.lv]` |

Baseline results (6 mm mesh, beat 1): EDV 94 mL, ESV 35 mL, LVEF 63%, peak pressure 13.7 kPa
(103 mmHg), EDP 1.1 kPa (8 mmHg).

## Running a simulation

Inside the physcardems Docker container (see the repository README for installation), from the
repository root:

```bash
python3 scripts/run_em_lvonly.py configs/lv_ellipsoid/baseline.toml
```

A one-beat run (`t_end_ms = 800`) takes about 1 to 1.5 hours on one laptop core; two beats about
2.5 to 3 hours. The first time you change an ion channel or calcium-sensitivity parameter, the run
first paces single cells to steady state, which adds a few minutes (once per parameter set).

To run without tying up your terminal, and to keep it from using every core:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup python3 scripts/run_em_lvonly.py configs/lv_ellipsoid/baseline.toml > runs/baseline.out 2>&1 &
tail -f runs/baseline.out
```

Each line of the log is one mechanics step: time, maximum active tension, fibre stretch range,
cycle phase, LV volume, pressure and outflow, base motion, solver iterations and wall time.

## How the code works

`scripts/run_em_lvonly.py` is short on purpose: its `main()` is the simulation in outline.

```python
sim = load_settings(config_path)                  # config + case -> one settings object
geo = setup_geometry(sim)                         # mesh, fibres, surfaces, cavity volume
ep = setup_electrophysiology(sim, geo)            # cell model at every node, activation times, monodomain
mech = setup_mechanics(sim, geo)                  # passive tissue + active tension, boundaries, cavity
cycle = setup_circulation(sim, mech)              # five-phase cardiac cycle + Windkessel
out = setup_outputs(sim, geo, ep, mech)           # log, figures, ParaView file, checkpoints
t_start = initialise(sim, ep, mech, cycle, out)

for t in time_steps(sim, t_start):                # one mechanics step at a time
    ep.advance_to(t)                              # electrophysiology substeps up to t
    transfer_ep_to_mechanics(ep, mech)            # crossbridge states -> active tension
    if not cycle.step(t):                         # solve mechanics with this phase's constraint
        break
    transfer_mechanics_to_ep(mech, ep)            # fibre stretch -> cell model
    out.record(t, ep, mech, cycle)
out.finish()
```

Each `setup_...` function lives in `src/physcardems/lv_simulation.py`, in the same order, so
you can read the detail of any step there. Electrophysiology and mechanics take turns each time
step: the cell model produces crossbridge states that set the active tension, the mechanics
solve deforms the tissue, and the resulting fibre stretch feeds back into the cell model.

## Outputs

In `runs/lv_ellipsoid/<config name>/`:

| File | Contents |
|---|---|
| `summary.png` | Pressure, volume, PV loop (with EF, EDV, ESV, pressures), outflow, active tension, det F, fibre stretch, 12-lead pseudo-ECG |
| `metrics.json` | Biomarkers for each beat: EDV, ESV, SV, LVEF, EDP, peak pressure, peak ejection rate, peak dP/dt |
| `log.csv` | One row per mechanics step; columns described in `src/physcardems/plotting_lv.py` |
| `pseudo_ecg.csv`, `pseudo_ecg.png` | 12-lead pseudo-ECG |
| `simulation.bp` | Fields for ParaView (open the folder with the ADIOS2 VTX reader): displacement `u`, voltage, calcium, active tension, fibre stretch, cell type, activation time. Use Warp By Vector on `u` to see the deformation |
| `active_stats.csv` | Active tension and fibre stretch per cell type over time |
| `config_used.json` | The exact settings of this run, with the git commit |
| `checkpoints/` | Saved states for restarting |

Biomarkers are read from phase changes: EDV at the start of isovolumic contraction, ESV at the
start of isovolumic relaxation.

## Changing parameters

Copy the baseline config, edit the copy, and run it:

```bash
cp configs/lv_ellipsoid/baseline.toml configs/lv_ellipsoid/my_run.toml
python3 scripts/run_em_lvonly.py configs/lv_ellipsoid/my_run.toml
```

The config is grouped by what you are likely to change. The main controls:

| To change | Config entry | Expected effect |
|---|---|---|
| Ion channel conductances | `[cell.ep_scales]`, e.g. `PCa_b` (L-type calcium), `GKr_b` (hERG) | APD, QT, calcium transient; PCa_b also contraction strength |
| Peak active tension | `[land.scales] Tref` | Contraction strength: higher -> smaller ESV, higher LVEF |
| Calcium sensitivity | `[land.overrides] cat50_ref` | Lower -> more tension at the same calcium |
| Passive stiffness | `[mechanics.material] stiffness_scale` | Lower -> more filling (larger EDV) |
| Epicardial stiffness | `[mechanics.boundary] epi_spring_pa_per_m` | Higher -> less whole-heart motion, smaller EDV |
| End-diastolic pressure | `[circulation.lv] p_end_diastole_pa` | Higher -> larger EDV |
| Afterload | `[circulation.lv.windkessel]` | `p_init_pa`, `characteristic_impedance...` up or `compliance...` down -> higher peak pressure; `resistance...` is peripheral resistance |
| Heart rate | `[time] pcl_ms` | Cycle length; changes the steady state |

These directions come from parameter sweeps on this model (see `configs/lv_ellipsoid/NOTES.md`).
Change one thing at a time when you are learning how the model behaves.

## Comparing runs

```bash
python3 scripts/compare_meshes_lv.py runs/lv_ellipsoid/baseline runs/lv_ellipsoid/my_run --out compare.png
```

prints volumes and pressures at matching times and the biomarkers of each completed beat, and
overlays volume, pressure and the PV loops. `scripts/replot_summary_lv.py <run folder>` redraws
one run's `summary.png` and prints its beat-by-beat biomarkers.

## Many runs: sweeps

For populations of models (in silico trials, emulator training data), `scripts/sample_lhs_lv.py`
draws a Latin hypercube sample of configs around the baseline, `scripts/run_pool.sh` runs them
several at a time, and `scripts/collect_lv.py` tabulates the biomarkers with Spearman
correlations. Each config gets its own output folder. Ask before launching large sweeps on shared
machines.

## A new anatomy

Copy this folder, edit `anatomy.toml` (radii, wall thickness, base height, mesh size, fibre angles,
conduction velocities) and regenerate:

```bash
cp -r cases/lv_ellipsoid cases/my_lv
rm -r cases/my_lv/geometry
python3 scripts/make_lv_ellipsoid.py cases/my_lv/anatomy.toml
```

This writes the mesh, fibres, activation map and `geometry_report.json` (volumes, wall thickness,
mass, helix angles). Point a config at the new case with `case = "../../cases/my_lv/case.toml"`.
The anatomy is the unloaded shape, so the cavity fills to a larger end-diastolic volume.

## Known limitations

- Peak ejection rate (about 1000 mL/s) is well above measured values (300 to 500 mL/s), a
  difficulty shared with other whole-heart models.
- Peak pressure (13.7 kPa) is slightly below the 14.3 to 15.3 kPa target.
- The 6 mm mesh reads about 2 percentage points lower on LVEF than finer meshes; use it for
  comparisons between runs, not as an absolute value.
- The activation sequence is prescribed (not computed from conduction), and the Land tension uses a
  smooth activation-dependent reference tension (`[land.tref_sigmoid]`) to keep resting tension low.
  Both are modelling choices worth stating in any write-up.
- Passive stiffness is 60% of the eLife values, chosen so this geometry fills to a physiological EDV.