# Hearing project: assembling the nine model variants A–I - Jake Abraham
"""
PLAIN WORDS
-----------
One place that wires the parts together into each model we compare (see config.VARIANTS):

  A  NP            nothing — the corrupted audio as-is
  B  MCTR          old-school detector, no learning
  C  RNN-3         Keshavarzi's network exactly (3-frame memory)
  D  RNN-stream    same network but with continuous memory        <- "is it just more context?"
  E  RNN-big       D, made as big as the proposed model           <- "is it just more parameters?"
  F  DBN-replace   belief network + attention only, no direct filterbank input (the original proposal)
  G  Hybrid-noattn filterbank + belief network, attention off
  H  Hybrid        filterbank + belief network + speech attention  <- PROPOSED
  I  IRM           the perfect mask computed from the clean speech (ceiling)

Every variant exposes the same function: mixture audio in -> mask out (and audio out), so evaluation
code never needs to know which model it is scoring.

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
build(variant, key, dbn_params=None, memories=None) -> Model with
    .features(x) -> dict      # front end (+ DBN forward, + attention) — frozen parts only
    .mask(params, feats)      # trainable part
    .enhance(params, x) -> (y, M)

features(x) by variant:
    C, D, E:  fb = log_features(band_energies(stft(x)))
    F:        dbn = dbn_distributed.forward(dbn_params, fb, attn=attention_feedback, memories)
              (fb is computed but NOT passed to the LSTM — only to the DBN)
    G:        fb + dbn_distributed.forward(..., attn=None)
    H:        fb + dbn_distributed.forward(..., attn=attention_feedback, memories)
    B:        not a network: enhance = baseline_mctr.mctr
    A:        M = ones;  I: M = ideal_ratio_mask(s, v + n)  (needs the clean reference)

Sizing variant E:
    target = count_params(H mask net) + count_params(DBN)   # count the frozen DBN too: it's real model size
    binary-search a width multiplier k so LSTM sizes (128k, 64k) give count_params(E) within 2% of target
    print both counts in the experiment log.

Precompute vs on-the-fly: during stage 4 the DBN is frozen, so DBN outputs can be cached per test
mixture; for training with dynamic mixing compute them on the fly (jit the DBN forward).
"""

from config import CFG, VARIANTS


def build(variant: str, key, dbn_params=None, memories=None):
    assert variant in VARIANTS, f"unknown variant {variant}"
    raise NotImplementedError


def size_matched_width(target_params: int) -> tuple:
    """Return LSTM sizes for variant E whose parameter count matches target_params within 2%."""
    raise NotImplementedError
