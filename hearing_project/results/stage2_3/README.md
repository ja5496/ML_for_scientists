# Stages 2–3 results — belief network pretraining + speech/transient memories (CPU, 2026-11-17)

Produced by `./run_pipeline.sh` (first two steps): `python dbn_pretrain.py` then `python run_experiments.py stage3`.
Total CPU time ≈ 2.2 h (one container restart mid-way; the run resumed from per-LSN checkpoints).

## Stage 2 — unsupervised pretraining of the 15-branch causal belief network

- Pretraining audio: LibriSpeech dev-clean (3 h speech) + FSD50K eval clips that carry no transient label
  (4 h everyday sounds, 2 h music). No VCTK, no test transients.
- Each LSN (48 units, 12-frame = 30 ms kernels at its own rate) trained for 10 epochs (4,910 steps) with CD-1 + sparsity.
- Held-out reconstruction error fell 2–100× per LSN; no dead units anywhere; firing rates 0.04–0.14 (target 0.05).
  Layer 1 (real-valued gammatone input) plateaus at ~0.5× its starting error, which is expected for a Gaussian visible layer.
- LSN11 templates (bottom of `diagnostics.png`) look like what auditory front ends learn: vertical onset/offset
  bars (click detectors) and horizontal harmonic stripes (voicing detectors).
- Speech-vs-transient firing similarity per LSN: cosine 0.47–0.75 (lower = more distinct) — every branch already
  responds differently to the two classes before any labels are used.

## Stage 3 — NMF memories (4 speech + 2 transient + 2 free bases per LSN) and go/no-go

Speech share of the belief, on held-out speech vs held-out transients:

| LSN | 11 | 21 | 22 | 31 | 32 | 33 | 34 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gap | .27 | .38 | .31 | .30 | .29 | .30 | .25 | .41 | .33 | .43 | .37 | .42 | .38 | .34 | .43 |

**15/15 sub-networks separate speech from transients (gap > 0.2; the plan needed ≥ 3) → GO.**
The slower, deeper branches (layer 4, up to ~0.45 s of context) separate best, which is the premise of the
project: longer context should help tell a crinkle from a consonant.

## Files

- `diagnostics.png`, `diagnostics.json` — training curves, firing similarity, LSN11 templates
- `pretrain_log.txt` — every evaluation step of every LSN
- `stage3_go_no_go.txt`, `memory_separability.json` — the memory check above
- `dbn/` — frozen weights used by variants F, G, H: `lsn{layer}{index}.pkl`, `feature_stats.npz`, `memories.pkl`
  (copy to `runs/dbn/` to reuse; `dbn_pretrain.load_dbn` reads that folder)
