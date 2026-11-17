# Hearing project: one causal convolutional RBM layer - Jake Abraham
"""
PLAIN WORDS
-----------
This is the building block of the belief network: a convolutional restricted Boltzmann machine
(CRBM), as in Bellur & Elhilali Eqs. 1–7. Think of it as a bank of learned "spectro-temporal
templates" (each a small picture spanning all frequency bands and ~30 ms of time). At every moment
each template reports how strongly the recent sound matches it (a probability between 0 and 1).

It is trained without labels: it learns templates that let it re-create the sound it sees, while
being forced to keep only ~5% of templates active at a time (sparsity), which makes the templates
specific and interpretable — like tuned neurons in auditory cortex.

Two changes from the paper:
1. Causal: a template at time t only looks at the past ~30 ms, never the future, so it can run inside
   a hearing aid with no added delay.
2. Pooling is causal too: when a layer runs at a slower rate (3x), each slow output summarises the
   last 3 fast frames and is only released after the 3rd one has arrived.

IMPLEMENTATION NOTES (status: IMPLEMENTED, tested in test_dbn.py)
-----------------------------------------------------------------
Shapes carry a batch dim: v (B, C_in, T); bases K (U, C_in, W); hidden bias d (U,); visible bias c (C_in,).
- Upward pass (Bellur Eq. 4) is a CAUSAL convolution: a[u, t] uses v[:, t-W+1 .. t] only.
- Probabilistic max-pooling (Lee et al. 2009) over non-overlapping blocks of P frames: within a block the
  P detection units of one template compete with an "off" state (softmax over [0, a_1..a_P]).
  Pooled output = P(any unit in the block on) = S / (1 + S), S = sum_k exp(a_k). P = 1 reduces to sigmoid(a).
  Pooled frame m covers detection frames mP .. mP+P-1, so it exists only once frame mP+P-1 has arrived.
- Training: contrastive divergence (CD-1) via autodiff of the free-energy difference F(v0) - F(v1) with v1
  held fixed (this has exactly the CD gradient), + Bellur's sparsity penalty lambda * sum_u (s - q_u)^2
  (Eq. 7, s = 0.05, lambda = 5) + small L2 on the bases. Optimiser: Adam (optim.py).
- Visible units: Gaussian (unit variance, inputs z-scored) for the first LSN, Bernoulli for the others
  (their inputs are the parent's pooled probabilities). Bellur's text says Bernoulli in Sec. II and
  Gaussian in Sec. V for the upper layers; Bernoulli is the natural fit for inputs in [0, 1].
- The top-down pass (only used inside CD training) looks at "future" hidden units; it is never used at
  inference, so the deployed network stays causal.
- Not implemented: the sparse convolutional autoencoder fallback (CFG.dbn.backend); add only if CD fails.
"""

import jax
import jax.numpy as jnp

from config import CFG

DN = ("NCH", "OIH", "NCH")          # (batch, channels, time) layout for lax.conv_general_dilated


def init_crbm(key, c_in: int, n_units: int = CFG.dbn.units_per_lsn, width: int = CFG.dbn.basis_width) -> dict:
    """Bases ~ N(0, 0.01); hidden bias starts at the sparsity target's log-odds (so ~5% of units fire)."""
    s = CFG.dbn.sparsity_target
    return {"K": 0.01 * jax.random.normal(key, (n_units, c_in, width)),
            "d": jnp.full((n_units,), jnp.log(s / (1 - s))),
            "c": jnp.zeros((c_in,))}


def hidden_preact(params: dict, v: jnp.ndarray) -> jnp.ndarray:
    """Causal convolution. v (B, C_in, T) -> a (B, U, T)."""
    W = params["K"].shape[-1]
    vp = jnp.pad(v, ((0, 0), (0, 0), (W - 1, 0)))
    return jax.lax.conv_general_dilated(vp, params["K"], (1,), "VALID", dimension_numbers=DN) + params["d"][None, :, None]


def _blocks(a: jnp.ndarray, pool: int) -> jnp.ndarray:
    B, U, T = a.shape
    T_p = T // pool
    return a[..., : T_p * pool].reshape(B, U, T_p, pool)


def _block_logsumexp(a: jnp.ndarray, pool: int) -> jnp.ndarray:
    """log(1 + sum_k exp(a_k)) per block, shape (B, U, T // pool)."""
    ab = _blocks(a, pool)
    return jax.nn.logsumexp(jnp.concatenate([jnp.zeros_like(ab[..., :1]), ab], axis=-1), axis=-1)


def prob_max_pool(a: jnp.ndarray, pool: int) -> tuple:
    """Return (detection_probs (B, U, T_p * pool), pooled_probs (B, U, T_p)). Causal blocks."""
    lse = _block_logsumexp(a, pool)                                       # log(1 + S), S = sum_k exp(a_k)
    ab = _blocks(a, pool)
    det = jnp.exp(ab - lse[..., None]).reshape(ab.shape[0], ab.shape[1], -1)
    return det, -jnp.expm1(-lse)                                          # S / (1 + S) = 1 - 1 / (1 + S)


def encode(params: dict, v: jnp.ndarray, pool: int) -> jnp.ndarray:
    """Deterministic upward pass used at inference: (B, C_in, T) -> pooled probabilities (B, U, T // pool)."""
    return -jnp.expm1(-_block_logsumexp(hidden_preact(params, v), pool))   # P(block on) = S / (1 + S)


def sample_hidden(key, a: jnp.ndarray, pool: int) -> jnp.ndarray:
    """Sample detection units: per block, at most one unit on (categorical over 'off' + P positions)."""
    ab = _blocks(a, pool)
    logits = jnp.concatenate([jnp.zeros_like(ab[..., :1]), ab], axis=-1)
    idx = jax.random.categorical(key, logits, axis=-1)
    onehot = jax.nn.one_hot(idx, pool + 1)[..., 1:]
    return onehot.reshape(ab.shape[0], ab.shape[1], -1)


def visible_mean(params: dict, h: jnp.ndarray, visible_type: str) -> jnp.ndarray:
    """Top-down pass (CD training only). h (B, U, T) -> mean of v (B, C_in, T)."""
    K = params["K"]
    W = K.shape[-1]
    Kt = jnp.flip(jnp.transpose(K, (1, 0, 2)), axis=-1)                 # (C_in, U, W)
    hp = jnp.pad(h, ((0, 0), (0, 0), (0, W - 1)))
    m = jax.lax.conv_general_dilated(hp, Kt, (1,), "VALID", dimension_numbers=DN) + params["c"][None, :, None]
    return m if visible_type == "gaussian" else jax.nn.sigmoid(m)


def free_energy(params: dict, v: jnp.ndarray, visible_type: str, pool: int) -> jnp.ndarray:
    """Per-example free energy, shape (B,)."""
    hidden = jnp.sum(_block_logsumexp(hidden_preact(params, v), pool), axis=(1, 2))
    c = params["c"][None, :, None]
    if visible_type == "gaussian":
        vis = 0.5 * jnp.sum((v - c) ** 2, axis=(1, 2))
    else:
        vis = -jnp.sum(c * v, axis=(1, 2))
    return vis - hidden


def cd_loss(params: dict, v0: jnp.ndarray, key, visible_type: str, pool: int) -> tuple:
    """CD-1 surrogate loss (its gradient is the CD update) + sparsity + L2. Returns (loss, metrics)."""
    T = (v0.shape[-1] // pool) * pool
    v0 = v0[..., :T]
    a0 = hidden_preact(params, v0)
    h_sample = sample_hidden(key, a0, pool)
    v1 = jax.lax.stop_gradient(visible_mean(params, h_sample, visible_type))
    n_frames = v0.shape[0] * T
    cd = (jnp.sum(free_energy(params, v0, visible_type, pool)) -
          jnp.sum(free_energy(params, v1, visible_type, pool))) / n_frames
    det, pooled = prob_max_pool(a0, pool)
    q = det.mean(axis=(0, 2))                                            # mean firing per unit
    sparsity = CFG.dbn.sparsity_weight * jnp.sum((CFG.dbn.sparsity_target - q) ** 2)
    l2 = 1e-4 * jnp.sum(params["K"] ** 2)
    metrics = {"recon_mse": jnp.mean((v0 - v1) ** 2), "mean_act": q.mean(),
               "dead_frac": jnp.mean(q < 1e-3), "pooled_act": pooled.mean()}
    return cd + sparsity + l2, metrics
