# Prescribed-Ta studies: running notes

One entry per study: the question, the spec, and what it showed. Run outputs live in
runs/<study>/ (untracked); the spec and its seed regenerate the samples.

## ta_lhs (2026-10-20)
- Question: what Ta amplitude, timing and shape give a healthy LVEF on rodero_05 with the current mechanics and circulation?
- Spec: configs/ta_sweep/lhs_ta.toml. 5 parameters (Ta_peak x5 to 10, t_peak x0.8 to 1.2, a_rise x0.6 to 1.6, sigma_decay x0.8 to 1.2, Ta_d_frac 0 to 0.01), 128 samples, ARCHER2, 2 nodes x 64.
- Result: best LVEF ~52%, 10 of 128 in 51-70%. Amplitude sets ESV (Spearman -0.99) but with low gain (ESV ~70 to 58 mL over 5 to 10x). Ta_d reduces filling (EDV -0.95, SV -0.91). t_peak and a_rise lower peak ejection rate (-0.56, -0.31) without changing SV. sigma_decay has no effect. Peak ejection rate 850-1280 mL/s (target 308-512); RVEF 20-25% with RV underfilled (EDV 72-78 mL).

## ta_lhs2 (2026-10-20)
- Question: can amplitude and a slower upstroke, with LV afterload, a separate RV amplitude and RV preload, reach the LV and RV criteria with matched stroke volumes?
- Spec: configs/ta_sweep/lhs_ta2.toml. 7 parameters (Ta_peak x6 to 14, t_peak x0.9 to 1.6, a_rise x0.6 to 2.5, rv_Ta_scale 0.5 to 1.5, LV R 0.7 to 1.4, LV R_c 0.03 to 0.12 mmHg s/mL, RV EDP 330 to 900 Pa); Ta_d fixed at cell values; runs stop 20 ms after both ventricles enter IVR.
- Result:

## Summary from collection of sweep outputs
 singularity exec -B "$PCE_WORK" "$PCE_IMAGE" "$PCE_VENV/bin/python3"   scripts/collect_sweep.py runs/ta_lhs --top 5 --summary
runs/ta_lhs: 128 samples, 128 complete

- runs/ta_lhs: 128 samples, 128 complete
- Targets: EDVL 88-161 mL, ESVL 31-68 mL, SVL 49-100 mL, LVEF 51-70 %, EDVR 85-166 mL, ESVR 27-77 mL, SVR 48-99 mL, RVEF 47-68 %, EDPL 0.5-1.5 kPa, PmaxL 14.3-15.3 kPa, PmaxR 2.8-3.6 kPa, dvdt_ejection 308-512 mL/s

Top five samples, ordered for robustness across the two importance profiles (equal weights,
and lv_first: LVEF/EDV/ESV, then LV pressures and ejection rate, then RV). Samples are sorted
first by how many profiles place them in their top 10, then by their worst rank across the
profiles. A sample that is best under one profile but poor under the other (e.g. sample 75:
1st by lv_first, 57th by equal weights) can therefore sit below samples that are moderate
under both. Use the rank_lv_first column for the LV-focused ordering.

| id | status | LVEF | EDVL | ESVL | SVL | PmaxL | EDPL | dvdt_ejection | RVEF | PmaxR | n_pass | n_criteria | score_equal | rank_equal | rank_lv_first |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 92 | complete | 49.5 | 124 | 62.8 | 61.6 | 15.2 | 1 | 991 | 23 | 3.64 | 6 | 12 | 4.32 | 9 | 24 |
| 8 | complete | 49.3 | 124 | 62.6 | 61 | 15.1 | 1 | 989 | 22.6 | 3.62 | 6 | 12 | 4.32 | 10 | 28 |
| 75 | complete | 51 | 121 | 59.2 | 61.7 | 15.5 | 1 | 1.09e+03 | 22.7 | 3.65 | 6 | 12 | 4.98 | 57 | 1 |
| 109 | complete | 46.6 | 125 | 66.9 | 58.5 | 14.9 | 1 | 941 | 22.3 | 3.64 | 6 | 12 | 4.26 | 8 | 62 |
| 32 | complete | 46.5 | 115 | 61.3 | 53.2 | 14.3 | 1 | 898 | 20.3 | 3.52 | 7 | 12 | 4.19 | 5 | 66 |

wrote runs/ta_lhs/results.csv

- Collected: 128 of 128 samples complete; 0 pass all 12 PV criteria.

| biomarker | target | min | median | max | n in range |
|---|---|---|---|---|---|
| EDVL (mL) | 88-161 | 102 | 120 | 127 | 128/128 |
| ESVL (mL) | 31-68 | 57.7 | 62.8 | 70.6 | 112/128 |
| SVL (mL) | 49-100 | 44.7 | 55.9 | 66.5 | 110/128 |
| LVEF (%) | 51-70 | 41.2 | 46.6 | 52.5 | 10/128 |
| EDVR (mL) | 85-166 | 73.2 | 74.8 | 78.5 | 0/128 |
| ESVR (mL) | 27-77 | 57.7 | 59.2 | 60.9 | 128/128 |
| SVR (mL) | 48-99 | 13.3 | 15.7 | 20.2 | 0/128 |
| RVEF (%) | 47-68 | 18.1 | 21.1 | 25.7 | 0/128 |
| EDPL (kPa) | 0.5-1.5 | 1 | 1 | 1 | 128/128 |
| PmaxL (kPa) | 14.3-15.3 | 13.6 | 15.2 | 17.8 | 62/128 |
| PmaxR (kPa) | 2.8-3.6 | 3.45 | 3.71 | 4.41 | 27/128 |
| dvdt_ejection (mL/s) | 308-512 | 777 | 1.07e+03 | 1.7e+03 | 0/128 |

- Best by lv_first: sample 75: LVEF 51.0%, EDV 121, ESV 59 mL, PmaxL 15.5 kPa, RVEF 22.7%; sample 48: LVEF 52.1%, EDV 124, ESV 59 mL, PmaxL 16.1 kPa, RVEF 24.1%; sample 76: LVEF 52.5%, EDV 127, ESV 60 mL, PmaxL 16.3 kPa, RVEF 25.7%