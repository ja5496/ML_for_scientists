# Stage 1 results — baseline reproduction (real data, first pass, 2026-11-17)

Produced by `python run_experiments.py stage1` on CPU (~45 min). One training seed; 100 test sentences
(20 unseen talkers) per (condition, STR) cell = 1,800 test mixtures per model.

- `stage1_results.png` — STOI, transient attenuation, speech preserved vs STR, per transient condition
- `summary.csv` — mean and 95% CI per model × condition × STR; `results_per_utterance.csv` — every mixture
- `go_no_go.txt` — stage-1 check (all PASS)
- `C/`, `D/` — trained weights (`best.pkl`, load with `model_variants.Trained.load`) and training curves
- `recipes.json`, `feature_stats.npz` — exact test mixtures and feature normalisation, for reproducibility
