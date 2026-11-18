# Stage 4 results — belief network + attention vs baselines (CPU first pass, 2026-11-18)

Produced in three steps:
1. `./run_pipeline.sh` (last step: `python run_experiments.py stage4`) trained and scored all nine models with seed 0.
2. `python run_experiments.py refresh4` added F and G to the trade-off sweep, the ablation tests and the
   equal-quality table, without retraining anything.
3. `python run_experiments.py multiseed` retrained C, E, G and H with seeds 1 and 2 (new initialisation **and** new
   training-data order), then scored them on the same test set. This took about 3 h of CPU.

**Setup:** each model trains for 2,000 steps × 16 two-second crops; within a seed, every model sees the same data stream.
Testing used 100 sentences from 20 unseen talkers in every (condition × STR) cell: 3 conditions × 6 STRs =
1,800 test mixtures per model. These are the **same mixtures for every model and every seed**, so all comparisons
are paired (per-mixture differences).

## Headline: 3 training seeds for C, E, G and H

Transient removed **at equal speech quality**: each model is made gentler or harsher (mask ** p) until all models
damage speech equally. Unseen transient types, STR −15 dB. Each cell is the mean over 3 seeds, with the lowest and highest seed in brackets:

| speech preserved → | 0.985 (gentle) | 0.975 | 0.965 (harsh) |
|---|---|---|---|
| C original RNN | 8.4 dB [7.9–8.8] | 10.8 [10.1–11.6] | 12.8 [11.9–13.7] |
| E size-matched LSTM | 8.7 [8.5–8.8] | 11.6 [11.5–11.7] | 14.1 [14.1–14.1] |
| G no attention | 8.8 [8.7–9.0] | 12.1 [11.8–12.5] | 15.0 [14.5–15.4] |
| **H proposed** | **9.3** [9.2–9.4] | **12.6** [12.3–12.9] | **15.4** [15.0–16.1] |
| **H − E**, per-seed gain | **+0.6** [+0.4 to +0.8] | **+1.0** [+0.6 to +1.3] | **+1.3** [+0.9 to +2.0] |
| H − C, per-seed gain | +0.9 [+0.6 to +1.3] | +1.7 [+1.4 to +2.2] | +2.6 [+2.3 to +3.1] |

Models as trained (no re-tuning). Each cell is a per-seed paired difference averaged over all 6 STRs of a condition, then
averaged over 3 seeds. Every seed has the same sign unless the cell says otherwise:

| | seen types | unseen types | crinkling |
|---|---|---|---|
| H − E, STOI | +0.0024 | +0.0036 | +0.0032 |
| H − E, transient removed | +0.5 dB | +1.0 dB | +0.5 dB |
| H − E, speech preserved | 0.000 (signs mixed) | −0.001 | 0.000 (signs mixed) |
| H − C, STOI | +0.0043 | +0.0047 | +0.0040 |
| H − C, transient removed | +1.7 dB | +2.3 dB | +0.3 dB (Wilcoxon p = 0.6, not significant) |

The spread across seeds is in `multiseed/multiseed_seeds.png` and `multiseed/multiseed_summary.txt`.

## All nine models, seed 0 only (averages over all 1,800 test mixtures)

| | STOI | transient removed | speech preserved |
|---|---|---|---|
| A no processing | 0.905 | 0.0 dB | 1.000 |
| B MCTR (non-neural) | 0.905 | 0.9 dB | 0.999 |
| C Keshavarzi RNN (original) | 0.925 | 11.6 dB | 0.976 |
| D same RNN, continuous memory | 0.925 | 11.9 dB | 0.976 |
| E LSTM size- & context-matched to H | 0.927 | 12.5 dB | 0.975 |
| F belief net + attention, **no filterbank** | 0.919 | 9.8 dB | 0.976 |
| G filterbank + belief net, **no attention** | 0.928 | 13.5 dB | 0.973 |
| **H proposed (filterbank + belief net + attention)** | **0.929** | 12.9 dB | 0.975 |
| I ideal mask (ceiling) | 0.980 | 22.2 dB | 0.976 |

Seed 0 at equal speech quality (unseen, −15 dB; speech preserved 0.985 / 0.975 / 0.965):
- C: 8.0 / 10.1 / 11.9 dB
- D: 8.2 / 11.0 / 13.5 dB
- E: 8.8 / 11.7 / 14.1 dB
- F: 6.5 / 8.6 / 10.6 dB
- G: 8.8 / 11.8 / 14.5 dB
- H: 9.2 / 12.3 / 15.0 dB

## What the numbers say

1. **H beats the original RNN (C) on every seed.** H's STOI is +0.004–0.005 higher in every condition. It removes
   +1.7 dB (seen types) and +2.3 dB (unseen types) more transient.
   At equal speech quality the gain is +0.9 to +2.6 dB, averaged over seeds. Seed 0 alone gave +1.3 to +3.1 dB, which
   was optimistic: seed 0 was C's weakest seed. The largest gains are on unseen impact sounds (seed 0: knock +3.4 dB,
   cutlery +2.9 dB, hammer and thump +2.3 dB).
2. **Keeping the filterbank is essential (H vs F, seed 0).** Without the raw 64 bands, the belief network alone is
   *worse* than the original RNN: −0.010 STOI and −2.5 to −3.6 dB versus H. This supports the "ear → cortex" design:
   the belief network adds to the filterbank and cannot replace it.
3. **H beats the fair control on every seed.** The fair control is an LSTM with the same size and the same memory (E),
   and this was the main question of the multi-seed check.
   - As trained, H removes +0.5 to +1.0 dB more transient for almost no speech cost (speech preserved changes by at
     most 0.001) and gains +0.002–0.004 STOI.
     This holds in all three conditions on all 3 seeds.
   - At equal speech quality the gain is +0.6, +1.0 and +1.3 dB, from the gentle to the harsh setting.
     The smallest single-seed gain is +0.4 dB.
   - E barely changes from seed to seed (11.5–11.7 dB at 0.975), so H's lead is not an unlucky E.
   - The seed-0 estimate (+0.4 to +0.85 dB) was, if anything, conservative.
4. **Both new parts contribute, but the belief network does so more reliably.**
   - **Belief-network features (G vs E):** +0.2, +0.5 and +0.9 dB at equal quality. G is never below E on any seed,
     but at the gentlest setting two seeds tie (+0.01 and +0.02 dB). The gain grows as the mask gets harsher. This corrects the seed-0 reading "G ≈ E", which came from G's weakest seed.
   - **Attention (H vs G):** +0.45 dB on average at equal quality, but not on every seed. Per-seed values range from
     −0.15 to +1.1 dB; seed 1 ties or slightly loses at the harsher settings.
   - **What attention reliably does, as trained, is make the model gentler.** On all 3 seeds it removes 0.8–1.2 dB less
     transient and keeps more speech (+0.004–0.005 speech preserved).
5. **Grandma's crinkling is still the easiest category, and the gain over the original RNN there is not significant.**
   C, E, G and H all remove 10–19 dB of it.
   - H vs C on crinkling: +0.3 dB (Wilcoxon p = 0.6) and +0.004 STOI.
   - H vs E on crinkling: +0.5 dB, positive on all 3 seeds.

   The clear gains are on impact sounds the model has never heard (knocks, slams, thuds).
6. **Plenty of headroom remains.** The ideal mask removes 22 dB at STOI 0.980; the proposed model reaches 12.9 dB and 0.929 (seed 0).

## Caveats (read before quoting any number)

- **Three seeds is still few.** The seed-level evidence is that every seed points the same way, not a formal test.
  The very small p-values in `multiseed/multiseed_summary.txt` (section C) treat the test mixtures as the unit, so they
  capture test-set noise but not training noise.
- F (no filterbank), D (continuous memory) and the nine-model table above are seed 0 only.
- The equal-quality frontier covers only one cell: unseen transients at −15 dB, 100 sentences.
- "Size-matched" means matched in parameter count (E has 745k trainable parameters ≈ H's 323k trainable + the frozen
  belief network). It is not matched in compute: H also runs a 15-branch convolutional network every 2.5 ms.
- The training budget is short (2,000 steps, about 4× Keshavarzi's). The belief network is frozen (no stage-5 fine-tuning yet).
- STOI is computed against clean speech at the input level. No hearing-loss simulation (HASPI) or listening tests yet.

## Files

- `../../figures/money_plot_results.png`: the headline figure. C, E and H are means over 3 seeds; thin lines show each seed.
- `stage4_ablation_frontier.png`: transient removed vs speech preserved for C, E, F, G, H (seed means; F has 1 seed).
- `stage4_results.png`: STOI, transient removed and speech preserved vs STR, for A, C, E, H, I in each condition (seed 0).
- `stage4_ablations.png`: the same for C, F, G, H, I (seed 0).
- `stage4_verdict.txt`: all paired tests for seed 0.
- `equal_speech_quality.csv`, `tradeoff_unseen_str-15.csv`: seed 0.
- `stage4_summary.csv`: mean ± 95% CI per model × condition × STR, seed 0.
- `stage4_results_per_utterance.csv`: 16,200 rows, seed 0.
- `multiseed/`: the multi-seed check.
  - `multiseed_summary.txt`: per-seed paired differences, the equal-quality table per seed, and seed-averaged tests.
  - `multiseed_seeds.png`: one marker per seed at speech preserved 0.975.
  - `multiseed_paired_diffs.csv`, `multiseed_equal_quality.csv`.
  - `seed{1,2}_results.csv` (per mixture), `seed{1,2}_tradeoff.csv`.
  - `weights/{C,E,G,H}/seed{1,2}/`.
  - `multiseed_log.txt`.
- `E/ F/ G/ H/`: seed-0 trained weights (`best.pkl`, load with `model_variants.Trained.load`) and training curves.
  F, G and H also need the frozen belief network in `../stage2_3/dbn/`.
