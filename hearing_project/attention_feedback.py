# Hearing project: online speech-directed attention (temporal coherence gate) - Jake Abraham
"""
PLAIN WORDS
-----------
This is Bellur & Elhilali's attention mechanism (Sec. IV-B, Eqs. 8–11), rebuilt so it can run live.

The idea from neuroscience ("temporal coherence"): when you attend to a voice, every neuron whose
activity rises and falls *in step* with the voice gets bound to it and amplified; neurons doing
their own thing get turned down. Here, at each moment and in each sub-network:

1. Ask "how much of what I'm seeing right now is speech, how much is transient, how much is other?"
   by explaining the current activity as a non-negative mix of the stored memories. This gives three
   live "belief" numbers: speech-belief, transient-belief, and leftover.
2. For every unit, keep a running (last ~200 ms) correlation between that unit's activity and the
   speech-belief. Units that move with speech are "coherent".
3. Coherent units pass; incoherent units are turned down (a smooth gate instead of Bellur's hard
   on/off, so it can be trained). The output is a blend: 70% original + 30% gated (Bellur's α=0.7).

Key change for our task: Bellur threw the turned-down activity away, because for classification you
only want the target. To *suppress* a crinkle you need to know where it is, so we keep the
turned-down part as a second "rejected" stream and hand both to the mask estimator, along with the
three belief numbers.

Everything here only uses the present and past, so it adds no delay.

IMPLEMENTATION NOTES (status: IMPLEMENTED, tested in test_attention.py)
-----------------------------------------------------------------------
Shapes carry a batch dim: F (B, U, T_native). Memories W = [W_speech | W_trans | W_free] (U, K), fixed.
1. beliefs: per-frame non-negative projection h = argmin_{h>=0} ||f - W h|| (projected gradient, 5 steps,
   step 1/L with L = largest eigenvalue of W^T W). beliefs = [sum h_speech, sum h_trans, relative residual].
   Each frame is independent -> causal by construction.
2. running coherence (online Eq. 9): exponentially-weighted Pearson correlation between each unit and the
   speech belief over the last ~CFG.attn.coherence_tau_ms, via jax.lax.scan (past frames only).
3. soft gate (Eq. 10): g = sigmoid((c - beta) / temperature); mix (Eq. 11): F_att = alpha F + (1 - alpha) g F.
4. F_rej = (1 - g) F is KEPT (Bellur discarded it) — it tells the mask estimator where the transient is.
alpha, beta, temperature are plain arguments (fixed in stage 4; learnable in stage 5).
"""

import jax
import jax.numpy as jnp
import numpy as np

from config import CFG

A = CFG.attn


def memory_matrix(memory: dict) -> tuple:
    """Stack a sub-network's memories -> (W (U, K), n_speech, n_trans)."""
    parts = [memory["W_speech"], memory["W_trans"]] + ([memory["W_free"]] if memory.get("W_free") is not None else [])
    W = jnp.asarray(np.concatenate(parts, axis=1))
    return W, memory["W_speech"].shape[1], memory["W_trans"].shape[1]


def project(F: jnp.ndarray, W: jnp.ndarray, n_iter: int = A.nnls_iters) -> jnp.ndarray:
    """Non-negative least squares per frame. F (B, U, T) -> H (B, K, T)."""
    WtW = W.T @ W
    L = jnp.linalg.eigvalsh(WtW)[-1] + 1e-6
    WtF = jnp.einsum("uk,but->bkt", W, F)
    H = jax.nn.relu(WtF) / (jnp.diag(WtW)[None, :, None] + 1e-6)
    for _ in range(n_iter):
        H = jax.nn.relu(H - (jnp.einsum("kj,bjt->bkt", WtW, H) - WtF) / L)
    return H


def beliefs(F: jnp.ndarray, mem: tuple) -> jnp.ndarray:
    """F (B, U, T) -> (B, 3, T): [speech belief, transient belief, relative residual]."""
    W, ns, nt = mem
    H = project(F, W)
    resid = jnp.linalg.norm(F - jnp.einsum("uk,bkt->but", W, H), axis=1) / (jnp.linalg.norm(F, axis=1) + 1e-6)
    return jnp.stack([H[:, :ns].sum(1), H[:, ns: ns + nt].sum(1), resid], axis=1)


def running_coherence(F: jnp.ndarray, H_att: jnp.ndarray, hop_ms: float, tau_ms: float = A.coherence_tau_ms) -> jnp.ndarray:
    """Causal exponentially-weighted correlation of each unit with the speech belief. -> (B, U, T) in [-1, 1]."""
    lam = float(np.exp(-hop_ms / tau_ms))
    B, U, T = F.shape

    def step(state, xs):
        mu_f, mu_h, var_f, var_h, cov = state
        f, h = xs                                                  # (B, U), (B,)
        mu_f = lam * mu_f + (1 - lam) * f
        mu_h = lam * mu_h + (1 - lam) * h
        df, dh = f - mu_f, (h - mu_h)[:, None]
        cov = lam * cov + (1 - lam) * df * dh
        var_f = lam * var_f + (1 - lam) * df ** 2
        var_h = lam * var_h + (1 - lam) * dh[:, 0] ** 2
        c = cov / jnp.sqrt(var_f * var_h[:, None] + 1e-8)
        return (mu_f, mu_h, var_f, var_h, cov), c

    z = jnp.zeros((B, U))
    init = (z, jnp.zeros(B), z, jnp.zeros(B), z)
    _, c = jax.lax.scan(step, init, (jnp.moveaxis(F, -1, 0), jnp.moveaxis(H_att, -1, 0)))
    return jnp.moveaxis(c, 0, -1)


def apply(F: jnp.ndarray, mem: tuple, hop_ms: float, alpha: float = A.alpha_mix, beta: float = A.beta_threshold,
          temperature: float = A.gate_temperature) -> tuple:
    """One LSN: F (B, U, T) -> (F_att, F_rej, beliefs (B, 3, T)). All causal."""
    bel = beliefs(F, mem)
    c = running_coherence(F, bel[:, 0], hop_ms)
    g = jax.nn.sigmoid((c - beta) / temperature)
    F_att = alpha * F + (1 - alpha) * g * F
    F_rej = (1 - g) * F
    return F_att, F_rej, bel


def make_attn(memories: dict, **kw):
    """Callable for dbn_distributed.forward(attn=...)."""
    mats = {lsn: memory_matrix(m) for lsn, m in memories.items()}
    return lambda F, lsn, hop_ms: apply(F, mats[lsn], hop_ms, **kw)
