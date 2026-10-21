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

## Summary from ta_lhs
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

## Summary from ta_lhs2

- runs/ta_lhs2: 128 samples, 127 complete, 1 not started
- Targets: EDVL 88-161 mL, ESVL 31-68 mL, SVL 49-100 mL, LVEF 51-70 %, EDVR 85-166 mL, ESVR 27-77 mL, SVR 48-99 mL, RVEF 47-68 %, EDPL 0.5-1.5 kPa, PmaxL 14.3-15.3 kPa, PmaxR 2.8-3.6 kPa, dvdt_ejection 308-512 mL/s

| id | status | LVEF | EDVL | ESVL | SVL | PmaxL | EDPL | dvdt_ejection | RVEF | PmaxR | n_pass | n_criteria | score_equal | rank_equal | rank_lv_first |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 122 | complete | 54.4 | 122 | 55.5 | 66.3 | 15.3 | 1 | 876 | 45.1 | 4.51 | 7 | 12 | 3.12 | 3 | 1 |
| 3 | complete | 51.5 | 127 | 61.4 | 65.1 | 16.1 | 1 | 918 | 25.1 | 3.67 | 6 | 12 | 4.52 | 9 | 2 |
| 90 | complete | 56.7 | 123 | 53.3 | 69.8 | 16.3 | 1 | 962 | 36.4 | 4.09 | 7 | 12 | 4.62 | 10 | 4 |
| 30 | complete | 55.1 | 123 | 55.4 | 67.9 | 16.3 | 1 | 970 | 31.9 | 3.89 | 7 | 12 | 4.7 | 11 | 6 |
| 102 | complete | 51.4 | 127 | 61.5 | 65.1 | 16.4 | 1 | 893 | 25.4 | 3.69 | 6 | 12 | 4.73 | 13 | 3 |
| 127 | complete | 52.7 | 123 | 58 | 64.6 | 16.8 | 1 | 889 | 31.2 | 3.87 | 7 | 12 | 4.77 | 14 | 7 |
| 24 | complete | 54.7 | 125 | 56.6 | 68.3 | 16.2 | 1 | 1.02e+03 | 39 | 4.2 | 7 | 12 | 4.8 | 15 | 8 |
| 108 | complete | 51.6 | 126 | 61 | 65 | 16.1 | 1 | 1.01e+03 | 30.2 | 3.93 | 6 | 12 | 4.96 | 18 | 5 |
| 55 | complete | 53 | 120 | 56.5 | 63.7 | 17 | 1 | 893 | 37.6 | 4.37 | 7 | 12 | 5.22 | 19 | 9 |
| 61 | complete | 54.8 | 126 | 57.2 | 69.2 | 16.9 | 1 | 934 | 21.9 | 3.48 | 7 | 12 | 5.56 | 22 | 10 |

wrote runs/ta_lhs2/results.csv

- Collected: 127 of 128 samples complete; 0 pass all 12 PV criteria.

| biomarker | target | min | median | max | n in range |
|---|---|---|---|---|---|
| EDVL (mL) | 88-161 | 119 | 124 | 127 | 127/127 |
| ESVL (mL) | 31-68 | 52.6 | 58 | 67.4 | 127/127 |
| SVL (mL) | 49-100 | 52.3 | 65.1 | 73.5 | 127/127 |
| LVEF (%) | 51-70 | 43.7 | 53.2 | 58.2 | 83/127 |
| EDVR (mL) | 85-166 | 78.4 | 89.5 | 100 | 89/127 |
| ESVR (mL) | 27-77 | 50.2 | 57.7 | 65.7 | 127/127 |
| SVR (mL) | 48-99 | 13.4 | 31.5 | 46.3 | 0/127 |
| RVEF (%) | 47-68 | 17 | 36.1 | 46.8 | 0/127 |
| EDPL (kPa) | 0.5-1.5 | 1 | 1 | 1 | 127/127 |
| PmaxL (kPa) | 14.3-15.3 | 14.6 | 20.3 | 28.9 | 3/127 |
| PmaxR (kPa) | 2.8-3.6 | 3.48 | 4.32 | 5.58 | 3/127 |
| dvdt_ejection (mL/s) | 308-512 | 586 | 912 | 1.42e+03 | 0/127 |

- Best by lv_first: sample 122: LVEF 54.4%, EDV 122, ESV 56 mL, PmaxL 15.3 kPa, RVEF 45.1%; sample 3: LVEF 51.5%, 
- EDV 127, ESV 61 mL, PmaxL 16.1 kPa, RVEF 25.1%; sample 102: LVEF 51.4%, EDV 127, ESV 61 mL, PmaxL 16.4 kPa, RVEF 25.4%