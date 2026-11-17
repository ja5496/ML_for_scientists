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

IMPLEMENTATION STATUS
---------------------
IMPLEMENTED: A (no processing), B (MCTR), C (RNN-3), D (RNN-stream), I (ideal mask).
STILL PSEUDOCODE: E, F, G, H — they need the pretrained belief network (stage 2–4) and H's size.

Shared interface: `process(variant, mix)` returns the processed mixture y AND the same processing applied
separately to the clean speech (s_hat) and to the transient (v_hat) — "shadow filtering" — so evaluation
can measure speech damage and transient reduction independently.

PSEUDOCODE FOR STAGE 4 VARIANTS (E, F, G, H)
-------------------------------------------
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

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import baseline_mctr
import frontend_gammatone as fe
import mask_estimator_lstm as ml
from config import CFG, VARIANTS

MODES = {"C": "window3", "D": "stream"}
NEURAL = set(MODES)
BUCKET = 8_000                     # pad utterances to multiples of 0.5 s so jit compiles only a few shapes


###########################################################
#                        FEATURES                         #
###########################################################

def features(x: jnp.ndarray, stats: dict) -> jnp.ndarray:
    """Waveform (B, N) -> z-scored log gammatone energies (B, T, 64): what the baseline RNN hears."""
    return fe.log_features(fe.band_energies(fe.stft(x)), stats)


def compute_feature_stats(batches, n_batches: int = 16) -> dict:
    """Per-band mean / std of log energies over training mixtures (never val/test)."""
    L = np.concatenate([np.asarray(fe.log_features(fe.band_energies(fe.stft(jnp.asarray(next(batches)["x"])))))
                        .reshape(-1, CFG.audio.n_bands) for _ in range(n_batches)])
    return {"mean": jnp.asarray(L.mean(0)), "std": jnp.asarray(L.std(0) + 1e-5)}


###########################################################
#                     TRAINED MODELS                      #
###########################################################

@dataclass
class Trained:
    variant: str
    params: dict
    stats: dict

    @property
    def mode(self):
        return MODES[self.variant]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        to_np = lambda t: jax.tree_util.tree_map(np.asarray, t)
        pickle.dump({"variant": self.variant, "params": to_np(self.params), "stats": to_np(self.stats)}, open(path, "wb"))

    @staticmethod
    def load(path: Path) -> "Trained":
        d = pickle.load(open(path, "rb"))
        to_j = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
        return Trained(d["variant"], to_j(d["params"]), to_j(d["stats"]))


@partial(jax.jit, static_argnums=(0,))
def _masks_jit(mode, params, stats, x):
    return ml.apply(params, features(x, stats), mode)


def neural_masks(model: Trained, xs: list, batch: int = 16) -> list:
    """Masks for a list of variable-length waveforms, batched by padded length."""
    order = np.argsort([len(x) for x in xs])
    out = [None] * len(xs)
    for i in range(0, len(xs), batch):
        idx = order[i: i + batch]
        L = int(np.ceil(max(len(xs[j]) for j in idx) / BUCKET) * BUCKET)
        xb = np.zeros((batch, L), np.float32)                        # fixed batch size -> fewer recompiles
        for r, j in enumerate(idx):
            xb[r, : len(xs[j])] = xs[j]
        M = np.asarray(_masks_jit(model.mode, model.params, model.stats, jnp.asarray(xb)))
        for r, j in enumerate(idx):
            out[j] = M[r, : fe.n_frames(len(xs[j]))]
    return out


###########################################################
#                PROCESS (with shadow filtering)          #
###########################################################

def _shadow(mix: dict, M) -> dict:
    run = lambda a: np.asarray(fe.enhance_with_mask(jnp.asarray(a), jnp.asarray(M)))
    return dict(y=run(mix["x"]), s_hat=run(mix["s"]), v_hat=run(mix["v"]), mask=np.asarray(M))


def process_batch(variant: str, mixes: list, model: Trained = None) -> list:
    """Apply one variant to a list of mixtures. Returns [dict(y, s_hat, v_hat, mask)]."""
    assert variant in VARIANTS
    if variant == "A":
        return [dict(y=m["x"], s_hat=m["s"], v_hat=m["v"], mask=None) for m in mixes]
    if variant == "B":
        res = []
        for m in mixes:
            g = baseline_mctr.detect_gains(m["x"])
            res.append(dict(y=baseline_mctr.apply_gains(m["x"], g), s_hat=baseline_mctr.apply_gains(m["s"], g),
                            v_hat=baseline_mctr.apply_gains(m["v"], g), mask=None))
        return res
    if variant == "I":
        return [_shadow(m, fe.ideal_ratio_mask(jnp.asarray(m["s"]), jnp.asarray(m["v"] + m["n"]))) for m in mixes]
    if variant in NEURAL:
        assert model is not None and model.variant == variant
        return [_shadow(m, M) for m, M in zip(mixes, neural_masks(model, [m["x"] for m in mixes]))]
    raise NotImplementedError(f"variant {variant} ({VARIANTS[variant][0]}) needs the belief network (stage 4)")


def size_matched_width(target_params: int) -> tuple:
    """LSTM sizes for variant E whose parameter count matches target_params within 2% (input = 64 bands)."""
    lo, hi = 0.25, 16.0
    for _ in range(60):
        k = (lo + hi) / 2
        sizes = (int(round(128 * k)), int(round(64 * k)))
        n = ml.count_params(ml.init_mask_net(jax.random.PRNGKey(0), CFG.audio.n_bands, sizes))
        lo, hi = (k, hi) if n < target_params else (lo, k)
    return sizes
