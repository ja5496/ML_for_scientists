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

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
Shapes: v (C_in, T) visible input; B (U, C_in, W) bases; d (U,) hidden bias; c (C_in,) visible bias.
U = CFG.dbn.units_per_lsn, W = CFG.dbn.basis_width.

hidden_preact(params, v):                         # Eq. 4, made causal
    v_pad = left-pad v with W-1 zeros along time  # NO right padding -> causal
    a[u, t] = sum_{c, r} B[u, c, r] * v_pad[c, t + r] + d[u]
    implement with jax.lax.conv_general_dilated (treat C_in as input channels, U as output channels)
    NOTE: jax conv is cross-correlation; that is fine (it's just the learned basis flipped).

prob_max_pool(a, pool):                           # Lee et al. 2009 probabilistic max-pooling, causal blocks
    if pool == 1: return sigmoid(a), sigmoid(a)
    reshape time into blocks of size `pool` (drop the trailing incomplete block during training;
      in streaming, a block is emitted only when complete)
    within each block, per unit:  P(h_k = 1) = exp(a_k) / (1 + sum_k' exp(a_k'))   (use logsumexp)
                                   P(pooled = 1) = 1 - 1 / (1 + sum_k exp(a_k))
    return h_probs (U, T), pooled_probs (U, T // pool)
    pooled frame m covers fast frames [m*pool, m*pool + pool - 1] -> available at fast frame m*pool + pool - 1

visible_mean(params, h):                          # Eq. 5 / 6 (top-down, used ONLY during CD training)
    v_mean[c, t] = sum_{u, r} B[u, c, r] * h[u, t - r + (W-1)] + c[c]   (transposed conv)
    gaussian visible (layer 1): return v_mean                (unit variance; inputs must be z-scored)
    bernoulli visible (layers >= 2): return sigmoid(v_mean)
    (This direction uses "future" hidden units — fine, it is never used at inference.)

free_energy(params, v, visible_type):
    a = hidden_preact(params, v)
    gaussian:  F = 0.5 * sum (v - c)^2 - sum softplus(a)
    bernoulli: F = -sum c * v           - sum softplus(a)

cd_loss(params, v0, key):                         # contrastive divergence via autodiff on free energy
    h0 = sigmoid(hidden_preact(params, v0)); h0_s = bernoulli(key, h0)
    v1 = visible_mean(params, h0_s)  (sample for bernoulli, mean for gaussian)
    v1 = stop_gradient(v1)
    loss = mean(free_energy(v0)) - mean(free_energy(v1))
    # Eq. 7 sparsity term:
    q = mean over batch and time of h0, per unit
    loss += CFG.dbn.sparsity_weight * sum_u (CFG.dbn.sparsity_target - q_u) ** 2
    + small L2 on B (1e-4)
    return loss, metrics(recon_mse = mean((v0 - v1)^2), mean_activation = mean(q))

train_step(params, opt_state, v_batch, key):  grads = jax.grad(cd_loss)(...); Adam update (reuse Lab_2 adam)
    Practical CRBM tips: small init (B ~ N(0, 0.01)), lr 1e-3 decayed, monitor recon_mse AND that
    mean activation hovers near 0.05; dead units (always 0) -> re-initialise them.

FALLBACK (CFG.dbn.backend == "sparse_conv_ae"):
    same encoder (causal conv + sigmoid + pooling), decoder = transposed conv,
    loss = MSE(v, decoder(h)) + sparsity KL(s || q). Same interfaces, so nothing else changes.

Unit tests:
    - causality: changing v[:, t0:] does not change hidden_preact[:, :t0]
    - pooled frame m is unchanged when fast frames >= (m+1)*pool are perturbed
    - on a toy dataset of shifted sinusoid chirps the learned bases look like chirp patches
"""

import jax
import jax.numpy as jnp

from config import CFG


def init_crbm(key, c_in: int, n_units: int = CFG.dbn.units_per_lsn, width: int = CFG.dbn.basis_width) -> dict:
    """Return {'B': (U, C_in, W), 'd': (U,), 'c': (C_in,)} with B ~ N(0, 0.01)."""
    raise NotImplementedError


def hidden_preact(params: dict, v: jnp.ndarray) -> jnp.ndarray:
    """Causal convolution. v: (C_in, T) -> (U, T)."""
    raise NotImplementedError


def prob_max_pool(a: jnp.ndarray, pool: int) -> tuple:
    """Return (hidden_probs (U, T), pooled_probs (U, T // pool)). Causal blocks."""
    raise NotImplementedError


def visible_mean(params: dict, h: jnp.ndarray, visible_type: str) -> jnp.ndarray:
    raise NotImplementedError


def free_energy(params: dict, v: jnp.ndarray, visible_type: str) -> jnp.ndarray:
    raise NotImplementedError


def cd_loss(params: dict, v0: jnp.ndarray, key, visible_type: str) -> tuple:
    raise NotImplementedError


def encode(params: dict, v: jnp.ndarray, pool: int) -> jnp.ndarray:
    """Deterministic upward pass used at inference: sigmoid probabilities, then causal pooling."""
    raise NotImplementedError
