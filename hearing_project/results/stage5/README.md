# Stage 5 results: end-to-end fine-tuning (CPU, 3 training seeds, 2026-11-18)

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

All three see the same new training crops in the same order (data seed + 100, so none of it was seen in stage 4).
Learning rates: 1e-4 for the mask network and the attention settings, 1e-5 for the belief network.
Each run keeps the checkpoint with the lowest validation error, including the starting point.
Scoring uses the same 1,800 test mixtures as stage 4, so all comparisons are paired.

## Headline: transient removed at equal speech quality

Unseen transient types, STR −15 dB. Each cell is the mean over 3 seeds. Brackets show the lowest and highest seed; in
the difference rows they show the three per-seed values in seed order:

| speech preserved → | 0.985 (gentle) | 0.975 | 0.965 (harsh) |
|---|---|---|---|
| E, 2,000 steps (stage 4) | 8.7 dB [8.6–8.8] | 11.6 [11.5–11.7] | 14.1 [14.1–14.1] |
| H, 2,000 steps (stage 4) | 9.3 [9.2–9.4] | 12.6 [12.3–12.9] | 15.4 [15.0–16.1] |
| E_cont, +2,000 steps | 9.3 [8.9–9.6] | 12.4 [12.0–13.0] | 15.2 [14.5–15.9] |
| H_ft, +2,000 steps, belief network fine-tuned | 9.7 [9.6–9.8] | 13.1 [12.9–13.3] | 16.0 [15.8–16.2] |
| **H_cont**, +2,000 steps, belief network frozen | **9.8** [9.6–10.0] | **13.3** [12.9–13.5] | **16.2** [15.8–16.5] |
| **H_ft − H_cont** (effect of fine-tuning) | −0.10 [−0.14, −0.13, −0.04] | −0.14 [−0.22, −0.19, +0.01] | −0.18 [−0.30, −0.28, +0.04] |
| **H_cont − E_cont** (H's lead after more training) | +0.55 [+0.36, +0.95, +0.34] | +0.82 [+0.54, +1.41, +0.52] | +1.06 [+0.68, +1.80, +0.69] |
| H − E at stage 4, for comparison | +0.60 [+0.42, +0.76, +0.63] | +0.96 [+0.61, +0.96, +1.30] | +1.34 [+0.85, +1.15, +2.01] |
| E_cont − E (extra steps alone) | +0.56 [+0.87, +0.33, +0.49] | +0.83 [+1.30, +0.41, +0.78] | +1.06 [+1.72, +0.42, +1.03] |
| H_cont − H (extra steps alone) | +0.51 [+0.81, +0.52, +0.20] | +0.69 [+1.23, +0.85, −0.00] | +0.78 [+1.55, +1.08, −0.29] |

Models as trained (no re-tuning): per-seed paired differences over all 6 STRs, averaged over 3 seeds. Every seed has
the same sign unless the cell says otherwise:

| | seen types | unseen types | crinkling |
|---|---|---|---|
| H_ft − H_cont, STOI | −0.0002 | −0.0005 | −0.0003 |
| H_ft − H_cont, transient removed | −0.1 dB (per seed −0.6, +0.3, +0.0) | −0.2 dB (per seed −0.8, +0.1, +0.1) | −0.1 dB (per seed −0.6, +0.2, +0.0) |
| H_cont − E_cont, STOI | +0.0023 | +0.0037 | +0.0033 |
| H_cont − E_cont, transient removed | +1.1 dB | +1.4 dB | +1.1 dB |
| H_cont − E_cont, speech preserved | −0.002 | −0.003 | −0.002 |

## What the numbers say

1. **End-to-end fine-tuning, as run here, does not help.**
   - At equal speech quality H_ft is slightly behind H_cont on seeds 0 and 1 (−0.13 to −0.30 dB) and ties on seed 2
     (−0.04 to +0.04 dB). It is never ahead by more than 0.04 dB.
   - As trained, its STOI is never higher than H_cont's in any condition on any seed (differences 0 to −0.001).
   - Validation error does not separate them either: 0.0184 vs 0.0186, 0.0188 vs 0.0187 and 0.0177 vs 0.0177 (H_ft vs
     H_cont, seeds 0–2).
   - **The likely reason is that this fine-tune hardly changed anything.** The belief network's weights moved by only
     0.2% of their overall size (largest single change 0.004). The attention settings barely moved: α 0.700 → 0.702–0.703,
     β −0.100 → −0.091 to −0.099, T 0.050 → 0.052–0.053.
   - So the test answers "does a gentle fine-tune help?" (no), not "can end-to-end training help?". With changes this
     small, the small deficit on seeds 0 and 1 may be training noise rather than harm. A stronger fine-tune (higher
     learning rates for the belief network and attention) is untested.
2. **H's lead over the size-matched LSTM survives more training, but shrinks a little.**
   - H_cont beats E_cont on every seed at every setting: +0.55 / +0.82 / +1.06 dB (smallest single seed +0.34 dB).
   - That is 0.05–0.3 dB less than the stage-4 lead on the same seeds (+0.60 / +0.96 / +1.34 dB), because the LSTM
     gained a little more from the extra steps than H did.
   - The interim two-seed reading ("the lead grows slightly") was reversed by seed 2.
   - As trained (averaged over seeds), H_cont removes +1.1 to +1.4 dB more transient and has +0.002–0.004 higher STOI,
     both positive on every seed, at a small speech cost (−0.002 to −0.003 speech preserved).
   - Part of H's advantage may be that it learns faster. Whether the lead persists at much longer training is the main
     open question for the headline claim.
3. **The models are still undertrained, but less uniformly than seed 0 suggested.**
   - The extra 2,000 steps gain +0.6 / +0.8 / +1.1 dB for E and +0.5 / +0.7 / +0.8 dB for H on average, about the same
     size as H's lead.
   - The gain varies a lot by seed. On seed 2, H barely improved: +0.2 dB at the gentle setting, −0.3 dB at the harsh one.
     That was the seed where stage-4 H was already strongest.
4. **Do not compare H_cont with the original RNN (C) from this table.** C did not get the extra 2,000 steps, so
   H_cont vs C would mix the architecture with the training budget.

## Caveats

- Three seeds is still few. The evidence is the sign agreement across seeds, not a formal test. The p-values in
  `stage5_summary.txt` (section C) treat test mixtures as the unit, so they capture test-set noise but not training noise.
- The equal-quality comparison covers only one cell: unseen transients at −15 dB, 100 sentences.
- The fine-tune used the learning rates set in the plan before any results (1e-5 for the belief network). It is one
  setting, not a search.

## Next steps this suggests

1. **Train longer** (C, E and H on the same extra data) to see whether H's lead over E and C holds, grows or shrinks
   toward convergence. This bears directly on the headline claim.
2. **A stronger fine-tune**, for example 10× the learning rate for the belief network and the attention settings. Only
   H_ft needs rerunning, because the controls exist: about 50 min of CPU per seed.
3. Tune attention (α, β) on validation data, then the harder crinkling test and HASPI (`PLAN.md` §9).

## Files

- `stage5_summary.txt`: validation error start → best and the learned attention settings per seed; all paired
  differences per seed (section A); the equal-quality table per seed (B); seed-averaged tests (C).
- `stage5_frontier.png`: transient removed vs speech preserved for all five models (seed means).
- `stage5_seeds.png`: one marker per seed at speech preserved 0.975.
- `stage5_paired_diffs.csv`, `stage5_equal_quality.csv`.
- `seed{0,1,2}_results.csv` (per mixture), `seed{0,1,2}_tradeoff.csv`: scores for H_ft, H_cont and E_cont. The
  stage-4 models' scores are in `../stage4/` (seed 0) and `../stage4/multiseed/` (seeds 1 and 2).
- `weights/{H_ft,H_cont,E_cont}/seed{0,1,2}/`: `best.pkl` (load with `model_variants.Trained.load`) and `log.csv`
  (training curve). H_ft's checkpoint contains its fine-tuned belief network and attention settings. H_cont still needs
  the frozen belief network in `../stage2_3/dbn/`.
- `stage5_log.txt`: the run log.

Reproduce: `python run_experiments.py stage5` (about 2 h 20 min of CPU per seed; finished runs are skipped).
`python run_experiments.py summary5` redraws the summary and figures from the saved scores.
