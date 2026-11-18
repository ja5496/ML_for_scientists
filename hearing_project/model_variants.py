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
IMPLEMENTED: all nine. A (no processing), B (MCTR), C (RNN-3), D (RNN-stream), I (ideal mask);
E (RNN-stream widened to H's total size: trainable mask net + frozen belief network);
F, G, H read the frozen belief network (runs/dbn/) and, for F and H, the memories + online attention.
Every learned variant is exposed as predict(params, aux, x) -> masks, where aux holds the frozen arrays.

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

import attention_feedback as af
import baseline_mctr
import dbn_distributed as dd
import frontend_gammatone as fe
import mask_estimator_lstm as ml
from config import CFG, VARIANTS

MODES = {"C": "window3", "D": "stream", "E": "stream", "F": "dbn", "G": "dbn", "H": "dbn"}
NEURAL = set(MODES)
DBN_VARIANTS = {"F": dict(fb=False, attn=True), "G": dict(fb=True, attn=False), "H": dict(fb=True, attn=True)}
K_SPEECH, K_TRANS = 4, 2                      # must match local_memory.K_SPEECH / K_TRANS
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
    extra: dict = None      # stage-5 fine-tuned models: their own {"dbn": ..., "attn": ...} replacing the frozen ones

    @property
    def mode(self):
        return MODES[self.variant]

    @property
    def aux(self):
        aux = build_aux(self.variant, self.stats)
        return {**aux, **self.extra} if self.extra else aux

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        to_np = lambda t: jax.tree_util.tree_map(np.asarray, t)
        d = {"variant": self.variant, "params": to_np(self.params), "stats": to_np(self.stats)}
        if self.extra:
            d["extra"] = to_np(self.extra)
        pickle.dump(d, open(path, "wb"))

    @staticmethod
    def load(path: Path) -> "Trained":
        d = pickle.load(open(path, "rb"))
        to_j = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
        return Trained(d["variant"], to_j(d["params"]), to_j(d["stats"]), to_j(d["extra"]) if d.get("extra") else None)


###########################################################
#            BELIEF-NETWORK FRONT END (F, G, H)           #
###########################################################

_FROZEN = {}
_DBN_DIR = None


def set_dbn_dir(path: Path):
    """Point every belief-network variant at a different trained DBN (e.g. the smoke-test one)."""
    global _DBN_DIR
    _DBN_DIR = Path(path)


def load_frozen(dbn_dir: Path = None) -> dict:
    """Frozen belief network + memories as a pytree of arrays (cached)."""
    from dbn_pretrain import DBN_DIR, load_dbn
    dbn_dir = Path(dbn_dir or _DBN_DIR or DBN_DIR)
    if dbn_dir not in _FROZEN:
        params, stats = load_dbn(dbn_dir)
        mem_path = dbn_dir / "memories.pkl"
        Ws = None
        if mem_path.exists():
            mem = pickle.load(open(mem_path, "rb"))
            Ws = {f"{g}{z}": jnp.asarray(af.memory_matrix(mem[(g, z)])[0]) for (g, z) in params}
        _FROZEN[dbn_dir] = {"dbn": {f"{g}{z}": p for (g, z), p in params.items()}, "dbn_stats": stats, "W": Ws}
    return _FROZEN[dbn_dir]


def attn_params_init() -> dict:
    """Unconstrained versions of the attention gate's settings, starting at the config values (stage 5 learns them)."""
    A = CFG.attn
    return {"alpha_logit": jnp.asarray(np.log(A.alpha_mix / (1 - A.alpha_mix)), jnp.float32),
            "beta": jnp.asarray(A.beta_threshold, jnp.float32),
            "log_temperature": jnp.asarray(np.log(A.gate_temperature), jnp.float32)}


def attn_settings(p: dict) -> dict:
    """alpha in (0, 1), temperature > 0."""
    return dict(alpha=jax.nn.sigmoid(p["alpha_logit"]), beta=p["beta"], temperature=jnp.exp(p["log_temperature"]))


def dbn_inputs(variant: str, aux: dict, x: jnp.ndarray) -> tuple:
    """Waveform (B, N) -> (fb (B,T,64) or None, Z (B,T,L,C), beliefs (B,T,L,3) or None). All causal."""
    from dbn_pretrain import dbn_features
    cfg = DBN_VARIANTS[variant]
    fb = features(x, aux["fb_stats"])
    T = fb.shape[1]
    params = {(int(k[0]), int(k[1:])): v for k, v in aux["dbn"].items()}
    attn = None
    if cfg["attn"]:
        kw = attn_settings(aux["attn"]) if "attn" in aux else {}     # stage 5: learned alpha / beta / temperature
        attn = lambda F, lsn, hop: af.apply(F, (aux["W"][f"{lsn[0]}{lsn[1]}"], K_SPEECH, K_TRANS), hop, **kw)
    out = dd.forward(params, dbn_features(x, aux["dbn_stats"]), attn)
    lsns = [l for l in dd.lsn_ids() if l in out]
    hold = lambda a, l: dd.hold_to_base(a, dd.rate_of(*l), T)
    if cfg["attn"]:
        Z = [jnp.concatenate([hold(out[l]["F_att_native"], l), hold(out[l]["F_rej_native"], l)], axis=1) for l in lsns]
        bel = jnp.stack([hold(out[l]["beliefs_native"], l) for l in lsns], axis=1)          # (B, L, 3, T)
        bel = jnp.concatenate([jnp.log1p(bel[:, :, :2]), bel[:, :, 2:]], axis=2)
        bel = jnp.transpose(bel, (0, 3, 1, 2))
    else:
        Z = [hold(out[l]["F_native"], l) for l in lsns]
        bel = None
    Z = jnp.transpose(jnp.stack(Z, axis=1), (0, 3, 1, 2))                                   # (B, T, L, C)
    return (fb if cfg["fb"] else None), Z, bel


###########################################################
#                 PREDICT / INIT per variant              #
###########################################################

def predict(variant: str, params: dict, aux: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Masks (B, T, 64) for any learned variant. aux = {"fb_stats", and for F/G/H "dbn", "dbn_stats", "W"}."""
    if MODES[variant] == "dbn":
        return ml.run_dbn_stream(params, *dbn_inputs(variant, aux, x))[0]
    return ml.apply(params, features(x, aux["fb_stats"]), MODES[variant])


def build_aux(variant: str, fb_stats: dict, dbn_dir: Path = None) -> dict:
    aux = {"fb_stats": fb_stats}
    if variant in DBN_VARIANTS:                          # E uses the frozen DBN only for sizing (init_params)
        aux.update(load_frozen(dbn_dir))
    return aux


def init_params(variant: str, key, dbn_dir: Path = None) -> dict:
    if variant in ("C", "D"):
        return ml.init_mask_net(key, CFG.audio.n_bands)
    fr = load_frozen(dbn_dir)
    n_lsn = len(fr["dbn"])
    if variant == "E":
        target = ml.count_params(init_params("H", key, dbn_dir)) + ml.count_params(fr["dbn"])
        return ml.init_mask_net(key, CFG.audio.n_bands, size_matched_width(target))
    cfg = DBN_VARIANTS[variant]
    c = CFG.dbn.units_per_lsn * (2 if cfg["attn"] else 1)
    return ml.init_dbn_mask_net(key, n_lsn, c, use_fb=cfg["fb"], use_beliefs=cfg["attn"])


@partial(jax.jit, static_argnums=(0,))
def _masks_jit(variant, params, aux, x):
    return predict(variant, params, aux, x)


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
        M = np.asarray(_masks_jit(model.variant, model.params, model.aux, jnp.asarray(xb)))
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
    raise ValueError(f"unknown variant {variant}")


def size_matched_width(target_params: int) -> tuple:
    """LSTM sizes for variant E whose parameter count matches target_params within 2% (input = 64 bands)."""
    lo, hi = 0.25, 16.0
    for _ in range(60):
        k = (lo + hi) / 2
        sizes = (int(round(128 * k)), int(round(64 * k)))
        n = ml.count_params(ml.init_mask_net(jax.random.PRNGKey(0), CFG.audio.n_bands, sizes))
        lo, hi = (k, hi) if n < target_params else (lo, k)
    return sizes
