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