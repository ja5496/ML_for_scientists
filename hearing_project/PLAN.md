# Hearing Project — Detailed Plan

**Goal:** Improve real-time transient noise reduction (TNR) for hearing aids by putting a
biologically inspired *distributed belief network + attention* stage (Bellur & Elhilali, 2020)
in front of the LSTM mask estimator of Keshavarzi, Reichenbach & Moore (2021).

This file is the overall plan. Every `.py` file in this folder has two layers:

1. **PLAIN WORDS** – a short explanation for a human reader (Jake).
2. **PSEUDOCODE** – structured, step-by-step instructions plus function stubs for a future
   implementing agent. Stubs raise `NotImplementedError` so nothing silently "works".

The code follows the style of `Lab_1.py` / `Lab_2.py`: JAX, pure functions, parameters in dicts.

---

## 0. What the two parent models actually are

| | Keshavarzi et al. 2021 (TNR RNN) | Bellur & Elhilali 2020 (DBN + attention) |
|---|---|---|
| Task | Regression: predict a time–frequency gain mask (IRM) | Classification: label a 3-s clip (10 UrbanSound classes) |
| Input | 64-channel gammatone energies, 5-ms frames, 2.5-ms hop | 128-channel auditory spectrogram (Chi et al.), 6-ms frames |
| Core | LSTM(128) → LSTM(64) → Dense(64, sigmoid) | Tree of 15 convolutional RBM sub-networks (LSNs), 300 units each, pooling η=1 (left) / τ=3 (right) |
| Context | **3 frames (~10 ms)**, causal | 36 ms (fastest) to ~1 s (slowest), **non-causal** |
| Attention | none | NMF "local memory" per class per LSN → temporal-coherence gate, **batch / non-causal** |
| Read-out | Mask × mixture, overlap-add resynthesis | Per-LSN **bidirectional** LSTM + fusion layer → softmax |
| Training | Supervised, MSE to IRM, Adam 1e-3, 5 epochs | DBN: unsupervised contrastive divergence; classifiers: supervised |
| Latency | ~7.5 ms (hearing-aid compatible) | not designed for real time |

## 1. Critical design decisions (and why)

### 1.1 The DBN cannot *replace* the filterbank — it must sit *after* it
The DBN's first layer takes a spectrogram as its input (Bellur Sec. II: "This setup takes as input
an auditory spectrogram"). It learns *spectro-temporal patterns over filterbank channels*, not
a filterbank itself. So "replace the gammatone filterbank with the DBN" is not literally possible.
The natural combination is the one Jake suggested as the alternative:

```
waveform → [EAR] gammatone filterbank → [A1/cortex] distributed belief network → [ATTENTION] speech-memory gate
                       │                                                               │
                       └──────────────── skip connection (raw filterbank energies) ────┤
                                                                                       ▼
                                                              [DECISION] LSTM mask estimator → IRM (64 bands)
                                                                                       ▼
                                                         mask × mixture STFT → overlap-add → output audio
```

We still test the "replace" idea as an ablation (variant F: no skip connection, LSTM sees only DBN
features), so Jake's original hypothesis gets a direct answer.

### 1.2 The mask has to stay in the filterbank domain
Bellur notes that reconstructing audio from DBN latents "is not straightforward". So the DBN
supplies *evidence* (features), but the network's *output* is still a 64-band gain in the same
domain as Keshavarzi's, which we know how to invert (gammatone-weighted STFT + overlap-add).

### 1.3 Everything must become causal (hearing-aid latency)
Bellur's model looks into the future in three places; each is redesigned:
- **Convolutions** → left-padded (causal) convolutions. Looking hundreds of ms into the *past* costs no latency (our slowest branch sees ~0.45 s; Bellur's ~1 s).
- **Pooling** → causal block pooling; a slow frame is released only after its last input frame.
  The slow output is then held (zero-order hold) until the next slow frame. *Off-by-one errors
  here leak future information and make results look falsely good — there is a unit test for this.*
- **NMF attention over the whole clip** → online version: per-frame nonnegative projection onto
  stored memories + exponentially-weighted running coherence.
- **BLSTM read-outs** → unidirectional (causal) LSTM.

Algorithmic latency stays at Keshavarzi's ~7.5 ms because only the 5-ms analysis window and the
overlap-add determine latency; extra *past* context is free.

### 1.4 Why the DBN could help at all (the scientific hypothesis)
Keshavarzi's LSTM sees only ~10 ms. In 10 ms a crinkle onset and a plosive ("p", "t", "k") can
look alike. The slow DBN branches (~100–450 ms of past context) carry a running "belief" about the ongoing
speech (pitch, formant trajectory, syllable rhythm). A transient is *temporally incoherent* with
that belief — exactly the cue Bellur's attention mechanism is built on. Hypothesis:
**long-range, multi-resolution context + a speech prior lets the network tell transients from
speech onsets, so it can suppress transients more while distorting speech less.**

### 1.5 Attention target is always "speech" — and we store a second memory for transients
Bellur attends to 1 of 10 classes. A hearing-aid wearer always wants speech, so the attended
memory is fixed (speech). Because we *also* know what we want to ignore, each LSN also stores a
"transient" memory. The online projection then explains every frame as speech + transient +
residual, which is a cleaner version of Bellur's Eq. 8 (where the non-target bases were free).

### 1.6 Don't let attention throw away the evidence the mask needs
Bellur's gate zeroes features incoherent with the target. For *classification* that's fine.
For *mask estimation* the network needs to know **where the transient is** in order to suppress
it. So the mask estimator receives three streams: raw filterbank energies, the attended
(speech-coherent) DBN features, and the rejected (speech-incoherent) DBN features.

### 1.7 Size: shrink the DBN to hearing-aid scale
Bellur: 15 LSNs × 300 units = 4,500 features/frame plus 15 BLSTM classifiers. Too big.
Plan: 15 LSNs × 32–64 units, each projected to 16 dims (240 total) before the LSTM. A
"Bellur-size" config is kept for an upper-bound experiment.

### 1.8 Fair comparisons (so the result is undeniable)
Two obvious objections, each with a control:
- "It's just more context." → **Variant D**: baseline LSTM that is stateful over the whole utterance.
- "It's just more parameters." → **Variant E**: baseline LSTM with parameters matched to the proposed model.
The proposed model must beat *both*, not just the original 3-frame RNN.

## 2. Data (all open-source)

**Important correction to the proposal.** Keshavarzi et al. did not release their transient-corrupted
mixtures or their 24 transient recordings. They used the open **VCTK** speech corpus and mixed in
transients themselves. We do the same with open transients. (The datashare link in the proposal
could not be opened from this sandbox; VCTK's handle is `10283/3443`. If that link is the
Valentini "noisy VCTK" set, note it contains *stationary* noises (DEMAND), not transients — useful
only for the background-noise stretch condition.)

| Role | Dataset | License | Use |
|---|---|---|---|
| Clean speech (mask training/test) | **CSTR VCTK** (110 English talkers, many accents) | CC BY 4.0 | Same corpus as Keshavarzi; disjoint talkers for train / val / test |
| Transients | **FSD50K** (Freesound, human-labeled, 200 classes) | CC0 / CC BY / CC BY-NC per clip | Classes such as *Crumpling and crinkling*, *Keys jangling*, *Knock*, *Door*, *Glass*, *Shatter*, *Dishes, pots and pans*, *Cutlery*, *Tap*, *Zipper*, *Scissors*. **Verify exact names in `vocabulary.csv`.** |
| Extra transients (optional) | **ESC-50** | CC BY-NC | door knock, glass breaking, can opening, mouse click, keyboard typing |
| DBN unsupervised pretraining | **LibriSpeech** (speech), **FSD50K** non-test classes (sounds), **MUSAN** music | CC BY 4.0 / mixed CC / CC BY 4.0 | Replaces Bellur's TIMIT + BBC SFX (not freely licensed) |
| Background-noise stretch test | **DEMAND** noise; babble built from held-out LibriSpeech talkers | CC BY-SA 3.0 / CC BY 4.0 | Keshavarzi's stated limitation #4 |
| Hearing-impaired metrics | **pyclarity** (Clarity Challenge toolkit: HASPI, HASQI, hearing-loss simulator) | MIT | Objective proxy for HI listeners |

**Splits (no leakage):**
- Talkers: 80 train / 10 val / 20 test (disjoint).
- Transient *classes*: ~70% train, ~30% held-out *unseen classes* (mirrors Keshavarzi's 15 train / 9 test).
  **Crumpling & crinkling is held out of training entirely** — it is the "grandma test".
- DBN pretraining data must exclude test talkers and test transient classes.

**Mixing recipe (matches Keshavarzi):** x(t) = s(t) + v(t). One transient per sentence at a random
position. STR = RMS(clean speech) / RMS(transient in a 5-ms window centered on its peak).
Train STR ∈ {−5, −10, −15} dB; test STR ∈ {−5, −10, −15, −20, −25, −30} dB.
Mixtures are generated on the fly (dynamic mixing) so every epoch sees new combinations.

## 3. Stages

| Stage | What | Output | Go / no-go check |
|---|---|---|---|
| 0 | Data download + mixing pipeline | `data_*.py` | Listen to 10 mixtures; STR computed correctly |
| 1 | **Reproduce Keshavarzi baseline** + MCTR + oracle IRM | Variants A, B, C, I | Baseline beats NP on STOI and falls short of IRM, like their Fig. 3. *If this fails, stop and fix before touching the DBN.* |
| 2 | Pretrain causal DBN (unsupervised, layer by layer) | `dbn_*.py` | Bases look like spectro-temporal patches; reconstruction error drops; rate–scale coverage like Bellur Fig. 4 |
| 3 | Estimate local memories (speech, transient) per LSN | `local_memory.py` | Speech memory activation H_att is high on speech frames, low on transient frames |
| 4 | Train mask estimator on frozen DBN + attention | `train_mask_estimator.py` | Beats variants C, D, E on validation |
| 5 | Optional end-to-end fine-tune (Bellur's stated future goal) | same | Small further gain, no overfitting |
| 6 | Evaluation, ablations, money plot | `evaluate_metrics.py`, `run_experiments.py`, `money_plot.py` | Money plot generated from real numbers |

## 4. Model variants

| ID | Name | Description |
|---|---|---|
| A | NP | No processing |
| B | MCTR | Non-neural multi-channel transient reduction (Keshavarzi Eqs. 3–4) |
| C | RNN-3 | Keshavarzi baseline: gammatone → LSTM128 → LSTM64 → Dense64, 3-frame windows |
| D | RNN-stream | Same as C but stateful over the utterance (controls for *context*) |
| E | RNN-big | Stateful, widened to match H's parameter count (controls for *size*) |
| F | DBN-replace | Jake's original idea: DBN + attention → LSTM, **no** raw-filterbank skip |
| G | Hybrid-noattn | Filterbank + DBN → LSTM (no attention) |
| **H** | **Hybrid (proposed)** | Filterbank + DBN + speech attention (attended + rejected streams) → LSTM |
| I | Oracle IRM | True IRM, upper bound |

## 5. Metrics

- **Intelligibility:** STOI / ESTOI (pystoi), HASPI (pyclarity) for simulated hearing loss.
- **Quality:** PESQ-WB, SI-SDR, HASQI.
- **Transient-specific (the comfort proxy):** apply the *same* predicted mask separately to s and v
  ("shadow filtering"). Then
  - *Transient attenuation (dB)* = energy of v before/after, inside the transient window.
  - *Speech preserved* = STOI(processed s, clean s) — 1.0 means speech untouched.
  These two numbers form the trade-off axes in the money plot.
- **Cost:** parameters, multiply-adds per frame, algorithmic latency (must stay ≤ ~10 ms), real-time factor.

## 6. Risks (ranked)

1. **The gain may be small in matched conditions.** Keshavarzi's RNN is already decent at −5 dB.
   Bellur's gains appeared mainly in *mismatched* conditions, so the expected win is on unseen
   transients, low STRs, and background noise. Design experiments to measure exactly there.
2. **Causality bugs** silently leak the future → falsely great results. Mitigation: a unit test that
   perturbs future input and asserts current output is unchanged.
3. **CRBM training is finicky** (contrastive divergence, sparsity). Fallback: a causal sparse
   convolutional autoencoder with the same tree topology (flag in `config.py`).
4. **Compute** of the DBN on a hearing-aid chip. Mitigation: slim config, report MACs, distill later.
5. **Subjective comfort** (the real endpoint) needs human listeners and IRB approval. This plan
   uses objective proxies only; a listening test is future work.

## 7. First pass (stage 0–1): what was actually built

**Data (downloaded 2026-11-17).**
- The proposal's datashare link *is* the VCTK 0.92 corpus. 2,000 sentences were read straight out of the
  11.7-GB zip (HTTP range requests): train 80 talkers × 20 = 1,600 (60% female, as in VCTK), val 10 × 10,
  test 20 × 15 = 300 (50/50). Utterances 001–024 (passages read by every talker) are excluded.
- FSD50K **eval** split only (first pass). 22 transient classes, each clip cut to 50 ms before / 450 ms after
  its peak, clips that aren't spiky (peak − median < 15 dB) or contain speech/music dropped.
  - seen (train/val/test): Chink_and_clink, Clapping, Crack, Dishes_and_pots_and_pans, Door, Drawer_open_or_close,
    Finger_snapping, Glass, Keys_jangling, Packing_tape_and_duct_tape, Scissors, Shatter, Tap, Typing, Zipper → 857 / 105 / 114 clips
  - unseen (test only): Coin_(dropping), Cutlery_and_silverware, Hammer, Knock, Slam, Thump_and_thud → 595 clips
    (caveat: this random split put most *impact* sounds in the unseen group)
  - grandma (test only): Crumpling_and_crinkling → 78 clips

**Choices the papers leave open (report these):**
- Window: sqrt-Hann analysis + synthesis (perfect reconstruction) instead of Keshavarzi's Hann analysis.
- Speech normalised to RMS 0.05 before mixing; STR uses Keshavarzi's 5-ms peak window (verified exact to 0.01 dB).
- Training: 2,000 steps × 16 two-second crops ≈ 4× Keshavarzi's frame budget, dynamic mixing, constant lr 1e-3.
- MCTR: running RMS = 50-ms one-pole smoother over past frames; bin→channel map {0,1},{2},{3,4},{5,6,7},{8..16}; Tukey 0.5.
- Evaluation: 100 of the 300 test sentences per (condition, STR) cell by default (`--n 300` for all).
- PESQ unavailable on this Python build; HASPI/HASQI not yet added.
