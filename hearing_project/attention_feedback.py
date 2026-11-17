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

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
Inputs per LSN: F (U, T_native) nonnegative activity; memories W_speech (U,Ks), W_trans (U,Kt), W_free (U,Kf);
hop_ms = 2.5 * rate of that LSN.

W = concat([W_speech, W_trans, W_free], axis=1)           # (U, K), K = Ks + Kt + Kf; FIXED (not learned in stage 4)

project_frame(f (U,), W) -> h (K,) >= 0:                   # online version of Eq. 8 (one column at a time)
    h = relu(W.T f) / (diag(W.T W))                         # warm start
    repeat CFG.attn.nnls_iters times:                       # projected gradient, step 1/L with L = ||W.T W||_2
        h = relu(h - (W.T (W h - f)) / L)
    return h
    -> vectorise over frames with vmap (no recurrence needed; each frame independent)

beliefs[t] = [sum(h[t, speech]), sum(h[t, trans]), ||f_t - W h_t|| / (||f_t|| + eps)]     # (3,)
H_att[t]   = beliefs[t, 0]                                                                    # Bellur's H_att

running coherence (online version of Eq. 9) via jax.lax.scan over t:
    lam = exp(-hop_ms / CFG.attn.coherence_tau_ms)
    state: mu_F (U,), mu_H, var_F (U,), var_H, cov (U,)
    mu_F  = lam mu_F + (1-lam) F[:, t];   mu_H = lam mu_H + (1-lam) H_att[t]
    dF = F[:, t] - mu_F;  dH = H_att[t] - mu_H
    cov   = lam cov   + (1-lam) dF * dH
    var_F = lam var_F + (1-lam) dF**2;  var_H = lam var_H + (1-lam) dH**2
    c[:, t] = cov / sqrt(var_F * var_H + eps)               # Pearson-like, in [-1, 1]
    (initial state: zeros; the first ~tau ms are a warm-up — fine for 2-s training crops,
     and in deployment the state just keeps running)

gate (soft version of Eq. 10):
    g = sigmoid((c - beta) / CFG.attn.gate_temperature)     # beta init = CFG.attn.beta_threshold
mix (Eq. 11):
    F_hat = g * F
    F_att = alpha * F + (1 - alpha) * F_hat                  # alpha init = CFG.attn.alpha_mix
    F_rej = (1 - g) * F                                      # NEW: the evidence for "not speech"
return F_att, F_rej, beliefs.T (3, T_native)

Learnable bits (stage 5 only, if CFG.attn.learn_alpha_beta): alpha (sigmoid-parameterised), beta, and
log-temperature per LSN. W stays fixed so the memories keep their meaning.

Sanity tests:
    - on clean speech, mean g > 0.7 at most LSNs; on an isolated transient, beliefs[1] >> beliefs[0]
    - perturbing F[:, t0:] leaves outputs[:, :t0] unchanged (causality)
    - alpha = 1 -> F_att == F exactly (attention switched off)
"""

import jax
import jax.numpy as jnp

from config import CFG


def project_frame(f: jnp.ndarray, W: jnp.ndarray, n_iter: int = CFG.attn.nnls_iters) -> jnp.ndarray:
    raise NotImplementedError


def running_coherence(F: jnp.ndarray, H_att: jnp.ndarray, hop_ms: float, tau_ms: float = CFG.attn.coherence_tau_ms) -> jnp.ndarray:
    raise NotImplementedError


def apply(F: jnp.ndarray, memory: dict, hop_ms: float, attn_params: dict = None) -> tuple:
    """Return (F_att, F_rej, beliefs) for one LSN. All causal."""
    raise NotImplementedError
