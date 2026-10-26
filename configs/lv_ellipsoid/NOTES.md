# LV ellipsoid (teaching case): running notes

One entry per investigation: the question, the spec, and what it showed. Run outputs live in
runs/ (untracked); configs and sampler specs regenerate them.

## Geometry (2026-10-21 to 23)
- Question: is the legacy simcardems ellipsoid usable for physiology, and if not, what replaces it?
- Legacy mesh (cases/lv_ellipsoid_legacy, converted by convert_legacy_ellipsoid.py): the Land et al. 2015
  benchmark ellipsoid (radii 7/10 mm short, 17/20 mm long, base 5 mm above the equator). 2.2 mL cavity,
  3 mm wall, 3.4 g, 207 vertices. Kept for comparison with legacy runs only.
- New case (cases/lv_ellipsoid, scripts/make_lv_ellipsoid.py from anatomy.toml): unloaded human-scale
  ellipsoid, r_short 21/30 mm, r_long 68/76 mm, base 8 mm above the equator. 73.1 mL cavity
  (analytic 73.8), 91.8 mL wall (96 g), cavity long axis 76 mm, mid-wall 8.7 mm, sphericity 0.50.
- Fibres: cardiac-geometriesx's default sign (endo +60, epi -60) gave a left-handed subendocardium.
  (-60, +60) gives the physiological right-handed endo / left-handed epi helix; checked in
  geometry_report.json (helix_angle_deg_right_handed_positive).
- Activation: graph-eikonal LAT with a fast endocardial layer (Purkinje stand-in), total 98 ms at 6 mm.
  Cell types by transmural bands 0.3 / 0.3 / 0.4.

## Resting active tension (2026-10-23)
- Question: why does the cavity shrink below its unloaded volume during preload?
- Observation (6 mm, Tref x7, no sigmoid): Ta max 22 kPa at t = 2 ms, before any activation (first LAT
  120 ms), rising to 42 kPa by 120 ms. Volume 73.1 -> 69.5 mL in the first step; EDV 70.4 mL at 1 kPa.
- Single cell (steady state ff00b0d2a086, isometric, paced at Tref x3): rest/peak Ta endo 6.49/256.7 kPa
  (2.5%), epi 5.06/258.9 (2.0%), mid 1.64/243.8 (0.7%). Diastolic Cai 0.119 / 0.113 / 0.085 uM.
  The prescribed-Ta BiV sweeps tolerated about 0.03% (Ta_d 0.06-0.10 kPa at peaks of 160-380 kPa).
  The cell fit without eLife overrides gave about 0.3%, so the eLife cat50_ref/kws changes look like
  the main source.
- Fix: sigmoidal Tref (src/physcardems/active.py, ZetaSplitSigmoidTref; [land.tref_sigmoid]).
  Tref_eff = Tref_rest + (Tref - Tref_rest) * sigma((n - n_half)/width), n = active fraction / rs.
  Mechanics-only, so no repacing. Phenomenological; not part of the published Land model.
- Check (scripts/check_tref_sigmoid.py, rest_scale 1, n_half 0.08, width 0.015): resting Ta endo
  15.1 -> 2.4 kPa, epi 11.8 -> 1.8, mid 3.8 -> 0.6; peaks unchanged (599 / 604 / 569 kPa); upstroke
  unchanged; late relaxation tail falls faster from about 450-500 ms.

## Mesh convergence (2026-10-23 to 24)
- Question: does the 6 mm student mesh give converged pressure-volume biomarkers?
- Spec: configs/lv_ellipsoid/baseline{,_h4,_h3,_h2}.toml, identical except mesh. Base held in x,
  epi spring 1e7 Pa/m, basal in-plane spring 1e6, sigmoid (rest_scale 1), volume-controlled filling.
- Result (beat 1): EDV 73.6 / 74.0 / 73.7 mL (6 mm / h4 / h3), within 0.5%. LVEF 57.3 / 59.1 / 59.3%.
  Pmax 13.2 / 13.7 / 13.7 kPa. Peak ejection rate 645 / 751 / 790 mL/s (less converged).
  h2 did not complete a beat. Step times about 6-8 / 21 / 33-40 / 80-150 s.
- Conclusion: h4 is converged for EDV, LVEF and Pmax; 6 mm reads about 2 points low on LVEF.
  Use h4 for reference results, 6 mm for quick exploration.
- Caveat: run with the old boundary conditions (fixed base, stiff spring), so convergence should be
  rechecked once the final BCs are set.

## lv_lhs1: contraction and circulation (2026-10-23)
- Question: which mechanics and circulation parameters drive the LV biomarkers on the ellipsoid?
- Spec: scripts/sample_lhs_lv.py, 96 samples, 8 factors: Tref scale 5-10, rest_scale 0.2-1,
  n_half 0.05-0.15, EDP 800-1500 Pa, p_init 8-11 kPa, R 0.9-1.3, C 0.8-2.5, R_c 0.02-0.06.
  Base held in x, epi spring 1e7, basal spring 1e6, volume-controlled filling. 6 mm, two beats.
- Result: EDV stuck at 73.6-74.7 mL in all 96 (below the 88 mL bound). EDP raised EDV by only about
  1 mL across its range (rho 0.83); rest_scale rho -0.53 over the same 1 mL.
  Tref drives ESV (-1.00), LVEF (0.99), peak ejection rate (0.98), dP/dt (0.99). LVEF 55-60% for most.
  Pmax driven by R_c (0.68), p_init (0.47), C (-0.35); many samples in 14.3-15.3 kPa.
  n_half has no effect (fix at 0.08). Peak ejection rate 583-718 mL/s, above target everywhere and
  tied to Tref as tightly as LVEF is (the LVEF vs ejection-rate conflict seen in Zingaro, Gerach).
  dP/dt max about 4000 kPa/s is an artefact of the pressure jump at the switch into IVC.
- Conclusion: contraction behaves; filling is blocked by the boundary conditions (next entry).

## lv_passive1: boundary conditions and filling (2026-10-23 to 26)
- Question: what limits filling, and which boundary conditions let the ellipsoid inflate?
- Diagnosis: with the base held in x and an epicardial spring, the wall can neither move outwards nor
  thin and lengthen towards the base, so the cavity cannot fill. In the BiV, filling works through
  wall thinning with the base moving up.
- Change: free base option ([mechanics.boundary] base = "free"). The cavity volume is measured from the
  current basal plane (mean basal displacement, lagged one step), since the open-base volume form is
  only exact with the cap at x = 0. Weak in-plane basal spring (1e4 Pa/m) removes rotation about the
  long axis. Base motion (mean and sd) logged each step.
- Spec: 48 samples, 4 factors: epi spring 1e5-1e8 Pa/m (log), material a-scale 0.3-1.5,
  rest_scale 0.2-1, EDP 800-1500 Pa. Free base, run to 130 ms (end of preload).
- Result: all 48 reached end-diastole. EDV 82.8-107.1 mL (most above 88). Base moves up 1.2-9.6 mm,
  sd 0.5-1.0 mm (planar enough except where dx is small: s039, s001, s017).
  Spearman with EDV: epi spring -0.73, EDP +0.43, a-scale -0.45, rest_scale -0.36.
- Best EDV near 105 mL at EDP about 1 kPa sits near the soft edges of the box (a-scale 0.32-0.64,
  spring 1e5-5e5), so higher EDV may need values outside the sampled range.
- Chosen for the student baseline (close to s021, away from the edges): free base, epi spring 2.5e5,
  basal spring 1e4, eLife HO a-values x 0.6, rest_scale 0.5, EDP 1.1 kPa. Expected EDV about 100-105 mL.

## Open items
- Filling: volume-controlled filling (0.046 mL/ms) starts while Ta is still above 100 kPa, needs about
  -25 kPa of suction, and refills too slowly (about 51 mL by 800 ms), so beat 2 starts with an
  artificial preload jump. filling_mode = "pressure" and "gain" are wired in run_em_lvonly.py; test
  pressure first, then the Alya relaxation gains if needed.
- dP/dt extraction: exclude the switch into IVC (or smooth) before using the dpdt_max criterion.
- Peak ejection rate above target at every LVEF in range; not addressable by the parameters so far.
- Recheck mesh convergence (6 mm vs h4) with the final boundary conditions.
- steady_state.py --config, so cell-parameter changes (e.g. PCa_b for the calcium channel blocker
  trial) can be paced into the case folder.
- MPI-parallel run_em_lvonly.py, for h2 and finer meshes.
- collect_lv.py: report EDV/EDP for runs that stop before ejection; --beat option (beat 2 is
  contaminated by the filling issue).

## Practical notes (ARCHER2)
- sbatch does not pass shell variables to the job here; use --export=ALL,PCE_SCRIPT=...,NPAR=...
- Run-script and thread selection: PCE_SCRIPT and PCE_THREADS in archer2/sweep.slurm and run_pool.sh.
- log.csv is rewritten every 10 mechanics steps, so a resubmitted run shows the old file for a while;
  delete stale run folders before resubmitting.
- pcepy (alias in .bashrc) runs Python in the container at the prompt; it does not expand inside
  Slurm scripts.