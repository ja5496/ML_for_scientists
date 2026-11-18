# Stage 4 results — belief network + attention vs baselines (CPU first pass, 2026-11-18)

Produced by `./run_pipeline.sh` (last step: `python run_experiments.py stage4`) and then
`python run_experiments.py refresh4`, which added F and G to the trade-off sweep, the ablation tests and the
equal-quality table without retraining anything.

**Setup:** one training seed per model, 2,000 steps × 16 two-second crops, the same data stream for every model.
Testing used 100 sentences from 20 unseen talkers in every (condition × STR) cell: 3 conditions × 6 STRs =
1,800 test mixtures per model, the **same mixtures for every model**. Statistics are paired (per-mixture differences, 95% CI and Wilcoxon p).

## Headline

| | STOI (all 1,800) | transient removed | speech preserved |
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

Transient removed **at equal speech quality**: each model is made gentler or harsher (mask ** p) until all
models damage speech equally. Unseen transient types, STR −15 dB:

| speech preserved → | 0.985 | 0.975 | 0.965 |
|---|---|---|---|
| C original RNN | 8.0 dB | 10.1 dB | 11.9 dB |
| D continuous memory | 8.2 | 11.0 | 13.5 |
| E size-matched LSTM | 8.8 | 11.7 | 14.1 |
| F no filterbank | 6.5 | 8.6 | 10.6 |
| G no attention | 8.8 | 11.8 | 14.5 |
| **H proposed** | **9.2** | **12.3** | **15.0** |

## What the numbers say

1. **H beats the original RNN (C).** It is +0.003–0.005 STOI better in every condition (p < 1e-11), and it removes
   +1.4 dB (seen) and +2.2 dB (unseen) more transient (p < 1e-12). On crinkling the gain is not significant; see point 5. At equal speech
   quality it removes +1.3 to +3.1 dB more. The gain is largest on unseen impact sounds: knock +3.4 dB,
   cutlery +2.9 dB, hammer and thump +2.3 dB.
2. **Keeping the filterbank is essential (H vs F).** Without the raw 64 bands, the belief network alone is *worse* than the
   original RNN: −0.010 STOI and −2.5 to −3.6 dB versus H. This supports the "ear → cortex" design: the belief network adds to the filterbank and cannot replace it.
3. **Against the fair control, a same-size LSTM with the same memory (E), the gain is small but consistent.**
   On unseen transients H gains +0.0026 STOI (p = 2e-14) and +0.7 dB (p = 2e-11). At equal speech quality the
   gain is +0.4 to +0.85 dB. On seen transients and on crinkling the gain is smaller (+0.2–0.3 dB, borderline).
4. **The attention gate makes the model gentler; at equal quality it adds about 0.45 dB.** Without attention (G), the model
   removes 0.4–0.8 dB *more* transient but damages speech more (p < 1e-9). On the frontier, H sits
   about 0.45 dB right of G. Without attention, the belief-network features roughly tie the size-matched LSTM (G ≈ E
   at equal quality). So the edge over E comes from the belief network and the attention working together.
5. **Grandma's crinkling is already the best-handled category, and it improved little.** C, E, G and H all remove
   10–19 dB of it. H vs C there is +0.25 dB, which is not significant, plus +0.003 STOI (p = 2e-12). Impact sounds the model has
   never heard (knocks, slams, thuds) are where the new model helps.
6. **Plenty of headroom remains.** The ideal mask removes 22 dB at STOI 0.980; the proposed model reaches 12.9 dB and 0.929.

## Caveats (read before quoting any number)

- **One training seed.** Differences of a few tenths of a dB between E, G and H could be seed noise. D and C,
  which differ only in memory type, differ by up to 1.7 dB on the frontier. Run seeds 1–2 before claiming H > E.
- The equal-quality frontier covers only one cell: unseen transients at −15 dB, 100 sentences. It has no CI yet.
- "Size-matched" means matched in parameter count (E has 745k trainable parameters ≈ H's 323k trainable + the frozen
  belief network). It is not matched in compute: H also runs a 15-branch convolutional network every 2.5 ms.
- The training budget is short (2,000 steps, about 4× Keshavarzi's). The belief network is frozen (no stage-5 fine-tuning yet).
- STOI is computed against clean speech at the input level. No hearing-loss simulation (HASPI) or listening tests yet.

## Files

- `stage4_results.png`: STOI, transient removed and speech preserved vs STR, for A, C, E, H, I in each condition
- `stage4_ablations.png`: the same for C, F, G, H, I
- `stage4_ablation_frontier.png`: transient removed vs speech preserved, for C, E, F, G, H
- `stage4_verdict.txt`: all paired tests; `equal_speech_quality.csv`; `tradeoff_unseen_str-15.csv`
- `stage4_summary.csv` (mean ± 95% CI per model × condition × STR); `stage4_results_per_utterance.csv` (16,200 rows)
- `E/ F/ G/ H/`: trained weights (`best.pkl`, load with `model_variants.Trained.load`) and training curves.
  F, G and H also need the frozen belief network in `../stage2_3/dbn/`.
- The money plot drawn from these numbers is `../../figures/money_plot_results.png`.
