# Stage 5 results: end-to-end fine-tuning (INTERIM, seed 0 only)

> **Interim.** Only training seed 0 is finished. Seeds 1 and 2 are still running on CPU, and this file will be rewritten
> when they finish. One seed is not enough to call any of the differences below; stage 4 showed that a single seed can
> mislead by about 0.5 dB.

## The question

In stage 4 the belief network (the 15-branch network trained without labels) and the attention settings were **frozen**.
Only the mask network on top of them was trained. Stage 5 asks whether training those parts too, end to end, helps.

A fair test needs controls, because any model improves with more training:

| Model | Starts from | Extra training | What is trained |
|---|---|---|---|
| **H_ft** | H (stage 4) | +2,000 steps on new data | mask network **+ belief network + attention** |
| **H_cont** | H (stage 4) | the same +2,000 steps | mask network only (belief network frozen, as in stage 4) |
| **E_cont** | E (stage 4) | the same +2,000 steps | the whole size-matched LSTM |

- **H_ft vs H_cont** isolates fine-tuning the belief network.
- **H_cont vs E_cont** checks whether H's lead over the size-matched LSTM survives more training.

All three see the same new training crops, in the same order (data seed + 100, so none of it was seen in stage 4).
Learning rates: 1e-4 for the mask network and attention, 1e-5 for the belief network.
Each run keeps the checkpoint with the lowest validation error, including the starting point.

## Seed 0

Transient removed **at equal speech quality** (unseen transient types, STR −15 dB):

| speech preserved → | 0.985 (gentle) | 0.975 | 0.965 (harsh) |
|---|---|---|---|
| E, 2,000 steps (stage 4) | 8.7 dB | 11.7 | 14.1 |
| H, 2,000 steps (stage 4) | 9.2 | 12.3 | 15.0 |
| E_cont, +2,000 steps | 9.6 | 13.0 | 15.9 |
| H_ft, +2,000 steps, belief network fine-tuned | 9.8 | 13.3 | 16.2 |
| **H_cont**, +2,000 steps, belief network frozen | **10.0** | **13.5** | **16.5** |
| **H_ft − H_cont** (effect of fine-tuning) | −0.1 | −0.2 | −0.3 |
| **H_cont − E_cont** (H's lead, after more training) | +0.4 | +0.6 | +0.7 |
| H − E in stage 4, same seed, for comparison | +0.4 | +0.6 | +0.9 |
| E_cont − E (effect of the extra steps alone) | +0.9 | +1.3 | +1.7 |

What seed 0 suggests (not yet established):

1. **Fine-tuning the belief network did not help.** H_ft is slightly *behind* H_cont at every quality setting
   (−0.1 to −0.3 dB). As trained, H_ft is the gentler model: it removes 0.6–0.8 dB less transient and keeps slightly more
   speech (+0.002 speech preserved) in all three test conditions.
   - Its validation error was the lowest of the three (0.0184 vs 0.0186 for H_cont), so the training objective and the
     test trade-off disagree here.
   - The attention settings barely moved: α 0.700 → 0.703, β −0.100 → −0.099, T 0.050 → 0.053.
2. **H's lead over the size-matched LSTM survives more training**, almost unchanged: +0.4 / +0.6 / +0.7 dB, against
   +0.4 / +0.6 / +0.9 dB at stage 4 on the same seed.
3. **Every model is still undertrained.** The extra 2,000 steps alone gain +0.9 to +1.7 dB (E) and +0.8 to +1.6 dB (H).
   That is about twice H's architectural lead. Longer training is the cheapest improvement available, and all
   comparisons so far are at a short budget.

## Files

- `stage5_summary.txt`: validation error start → best, the learned attention settings, all paired differences per seed
  (section A), the equal-quality table per seed (B), and seed-averaged tests (C).
- `stage5_frontier.png`: transient removed vs speech preserved for all five models.
- `stage5_seeds.png`: one marker per seed at speech preserved 0.975.
- `stage5_paired_diffs.csv`, `stage5_equal_quality.csv`.
- `seed0_results.csv` (per mixture), `seed0_tradeoff.csv`: scores for H_ft, H_cont and E_cont.
  The stage-4 models' scores are in `../stage4/`.
- `weights/{H_ft,H_cont,E_cont}/seed0/`: `best.pkl` (load with `model_variants.Trained.load`) and `log.csv` (training curve).
  H_ft's checkpoint includes its fine-tuned belief network and attention settings. H_cont still needs the frozen belief
  network in `../stage2_3/dbn/`.
- `stage5_log.txt`: the run log.

Reproduce: `python run_experiments.py stage5` (about 2 h 20 min of CPU per seed; finished runs are skipped).
`python run_experiments.py summary5` redraws the summary and figures from the saved scores.
