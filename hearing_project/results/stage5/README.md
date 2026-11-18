# Stage 5 results: end-to-end fine-tuning (INTERIM, seeds 0 and 1)

> **Interim.** Training seeds 0 and 1 are finished; seed 2 is still running on CPU, and this file will be rewritten
> when it finishes. Two seeds are not enough to call small differences; stage 4 showed that a single seed can mislead
> by about 0.5 dB.

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

## Seeds 0 and 1

Transient removed **at equal speech quality** (unseen transient types, STR −15 dB). Each cell is the mean of the two
seeds; the per-seed values are in brackets.

| speech preserved → | 0.985 (gentle) | 0.975 | 0.965 (harsh) |
|---|---|---|---|
| E, 2,000 steps (stage 4) | 8.6 dB | 11.6 | 14.1 |
| H, 2,000 steps (stage 4) | 9.2 | 12.4 | 15.1 |
| E_cont, +2,000 steps | 9.2 | 12.5 | 15.2 |
| H_ft, +2,000 steps, belief network fine-tuned | 9.8 | 13.2 | 16.2 |
| **H_cont**, +2,000 steps, belief network frozen | **9.9** | **13.4** | **16.4** |
| **H_ft − H_cont** (effect of fine-tuning) | −0.14 [−0.14, −0.13] | −0.21 [−0.22, −0.19] | −0.29 [−0.30, −0.28] |
| **H_cont − E_cont** (H's lead, after more training) | +0.65 [+0.36, +0.95] | +0.97 [+0.54, +1.41] | +1.24 [+0.68, +1.80] |
| H − E in stage 4, same two seeds, for comparison | +0.59 [+0.42, +0.76] | +0.79 [+0.61, +0.96] | +1.00 [+0.85, +1.15] |
| E_cont − E (effect of the extra steps alone) | +0.60 [+0.87, +0.33] | +0.85 [+1.30, +0.41] | +1.07 [+1.72, +0.42] |
| H_cont − H (effect of the extra steps alone) | +0.66 [+0.81, +0.52] | +1.04 [+1.23, +0.85] | +1.31 [+1.55, +1.08] |

What two seeds suggest (seed 2 still to come):

1. **Fine-tuning the belief network did not help.** At equal speech quality, H_ft is slightly *behind* H_cont on both
   seeds and at every setting (−0.13 to −0.30 dB), and the two seeds agree closely.
   - As trained, the two seeds sit at different operating points: on seed 0 H_ft is gentler than H_cont (0.6–0.8 dB less
     transient removed, +0.002 speech preserved); on seed 1 it is slightly harsher (+0.1 to +0.3 dB, −0.001 speech
     preserved). The equal-quality comparison removes that difference, and there H_ft is behind on both.
   - Its validation error matched H_cont's (seed 0: 0.0184 vs 0.0186; seed 1: 0.0188 vs 0.0187), so the training
     objective does not separate them either.
   - The attention settings barely moved: α 0.700 → 0.703, β −0.100 → −0.099 / −0.096, T 0.050 → 0.053 / 0.052.
2. **H's lead over the size-matched LSTM survives more training** and, on these two seeds, grows slightly: +0.65 / +0.97 /
   +1.24 dB, against +0.59 / +0.79 / +1.00 dB at stage 4. It is positive on both seeds at every setting.
3. **The models are still undertrained.** The extra 2,000 steps gain about +0.6 to +1.3 dB on average, comparable to H's
   lead from its design. For E the gain varies a lot between seeds (+0.9 to +1.7 dB on seed 0, only +0.3 to +0.4 dB on
   seed 1); for H it is steadier (+0.5 to +1.6 dB). All comparisons so far are at a short training budget.

## Files

- `stage5_summary.txt`: validation error start → best, the learned attention settings, all paired differences per seed
  (section A), the equal-quality table per seed (B), and seed-averaged tests (C).
- `stage5_frontier.png`: transient removed vs speech preserved for all five models.
- `stage5_seeds.png`: one marker per seed at speech preserved 0.975.
- `stage5_paired_diffs.csv`, `stage5_equal_quality.csv`.
- `seed{0,1}_results.csv` (per mixture), `seed{0,1}_tradeoff.csv`: scores for H_ft, H_cont and E_cont.
  The stage-4 models' scores are in `../stage4/`.
- `weights/{H_ft,H_cont,E_cont}/seed{0,1}/`: `best.pkl` (load with `model_variants.Trained.load`) and `log.csv` (training curve).
  H_ft's checkpoint includes its fine-tuned belief network and attention settings. H_cont still needs the frozen belief
  network in `../stage2_3/dbn/`.
- `stage5_log.txt`: the run log.

Reproduce: `python run_experiments.py stage5` (about 2 h 20 min of CPU per seed; finished runs are skipped).
`python run_experiments.py summary5` redraws the summary and figures from the saved scores.
