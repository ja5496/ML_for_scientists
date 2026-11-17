# Hearing project: the LSTM that decides the mask - Jake Abraham
"""
PLAIN WORDS
-----------
This is Keshavarzi et al.'s decision network, kept almost as-is: two LSTM layers (128 then 64 units)
and a final layer that outputs 64 numbers between 0 and 1 — one gain per frequency band per 2.5 ms.
A gain near 1 means "this is speech, keep it"; near 0 means "this is mostly transient, turn it down".

What changes is what it gets to look at. Besides the 64 filterbank energies it always had, it now
also receives, from each of the 15 belief sub-networks:
  - a compressed (16-number) summary of the speech-coherent and speech-incoherent activity, and
  - the three live belief numbers (speech / transient / leftover).

"Stream selection" (an idea Bellur & Elhilali proposed as future work): a tiny extra layer looks at
what the LSTM was thinking a moment ago and decides how much to listen to each sub-network right
now. It also tells us afterwards *which* sub-networks were useful for catching transients — a nice
interpretable result (the analogue of Bellur's Fig. 8).

Unlike Keshavarzi's version, which only saw 3 frames (~10 ms) at a time and then forgot everything,
this LSTM runs continuously and carries memory forward. (A baseline with that same change is
included as a control, so we can tell whether improvements come from the belief network or just
from the longer memory.)

IMPLEMENTATION STATUS
---------------------
IMPLEMENTED (used by the baselines C, D, E and later by F, G, H):
  - from-scratch LSTM cell (forget-gate bias 1.0), two layers (128, 64) + Dense(64) + sigmoid
  - mode "window3": Keshavarzi's 3-frame windows, memory reset for every output frame       (variant C)
  - mode "stream":  one continuous pass over the utterance, memory carried forward          (D, E, F, G, H)
  - loss: MSE between predicted mask and IRM (Keshavarzi's objective)
  The network takes a generic per-frame input vector u_t, so later variants only need to build u_t.
STILL PSEUDOCODE (stage 4, needs the pretrained belief network): per-LSN readouts, stream selection,
  belief numbers — see the block below. They will produce u_t and reuse everything implemented here.

PSEUDOCODE FOR STAGE 4 (belief-network inputs)
----------------------------------------------
init_mask_net(key, variant) -> params:
    in_fb   = 64 if variant uses filterbank skip else 0
    in_dbn  = 15 * readout_dim (16) if variant uses DBN else 0
    in_bel  = 15 * 3 if variant uses attention else 0
    readout[lsn]: Dense(2U -> 16) if attention (concat F_att, F_rej) else Dense(U -> 16)
    selector:     Dense(64 -> 15)   (reads previous LSTM-2 hidden state)   if use_stream_selection
    norm:         LayerNorm(in_fb + in_dbn + in_bel)
    lstm1: LSTM(in -> sizes[0]); lstm2: LSTM(sizes[0] -> sizes[1]); out: Dense(sizes[1] -> 64)
    (write the LSTM cell by hand with jax.lax.scan, consistent with Lab_2's from-scratch style;
     forget-gate bias init = 1.0)

step(params, carry, x_t) -> carry, mask_t:      # ONE 2.5-ms frame; used by lax.scan and by streaming
    fb_t, dbn_t (dict lsn -> (F_att_t, F_rej_t)), bel_t (15, 3) = x_t
    z_lsn = [readout[l](concat(F_att_t[l], F_rej_t[l])) for l in 15]       # each (16,)
    if stream selection:
        w = softmax(selector(carry.h2_prev))                                 # (15,), causal (previous state)
        z_lsn = [15 * w[l] * z_lsn[l]]
    u = norm(concat(fb_t, *z_lsn, bel_t.flatten()))
    h1, c1 = lstm1(u, carry.h1, carry.c1); h2, c2 = lstm2(h1, carry.h2, carry.c2)
    mask_t = sigmoid(out(h2))                                                # (64,)
    return new carry (store w for analysis), mask_t

apply(params, features, mode) -> masks (T, 64):
    mode == "stream": lax.scan over all T frames from a zero carry                (variants D, E, F, G, H)
    mode == "window3": for each j, run step over frames j-2, j-1, j from a ZERO carry,
                       keep the last output                                       (variant C, = Keshavarzi)

loss(params, batch):
    M = apply(...); IRM = ideal_ratio_mask(s, v + n)
    mse = mean((M - IRM)^2)                                                      # Keshavarzi objective
    optional (flag): + 0.5 * mean over frames where IRM < 0.5 of (M - IRM)^2      # emphasise transient frames
    return mse

count_params(params): used to size variant E to match H (see model_variants.py)
"""

import jax
import jax.numpy as jnp

from config import CFG


###########################################################
#                          LSTM                           #
###########################################################

def init_lstm(key, d_in: int, d_hidden: int) -> dict:
    """Glorot-uniform weights for [x, h] -> 4 gates (i, f, g, o); forget-gate bias = 1."""
    lim = jnp.sqrt(6.0 / (d_in + d_hidden + 4 * d_hidden))
    W = jax.random.uniform(key, (d_in + d_hidden, 4 * d_hidden), minval=-lim, maxval=lim)
    b = jnp.zeros(4 * d_hidden).at[d_hidden: 2 * d_hidden].set(1.0)
    return {"W": W, "b": b}


def lstm_cell(params: dict, x: jnp.ndarray, h: jnp.ndarray, c: jnp.ndarray) -> tuple:
    z = jnp.concatenate([x, h], axis=-1) @ params["W"] + params["b"]
    i, f, g, o = jnp.split(z, 4, axis=-1)
    c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, c


def init_mask_net(key, d_in: int, sizes: tuple = CFG.mask.lstm_sizes, d_out: int = CFG.mask.dense_out) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    lim = jnp.sqrt(6.0 / (sizes[1] + d_out))
    return {"lstm1": init_lstm(k1, d_in, sizes[0]), "lstm2": init_lstm(k2, sizes[0], sizes[1]),
            "out": {"W": jax.random.uniform(k3, (sizes[1], d_out), minval=-lim, maxval=lim), "b": jnp.zeros(d_out)}}


def _sizes(params):
    return params["lstm1"]["b"].shape[0] // 4, params["lstm2"]["b"].shape[0] // 4


def zero_carry(params, batch_shape):
    s1, s2 = _sizes(params)
    z = lambda n: jnp.zeros(batch_shape + (n,))
    return (z(s1), z(s1), z(s2), z(s2))


def step(params: dict, carry: tuple, u_t: jnp.ndarray) -> tuple:
    """ONE 2.5-ms frame for a batch: u_t (..., d_in) -> (new carry, mask_t (..., 64))."""
    h1, c1, h2, c2 = carry
    h1, c1 = lstm_cell(params["lstm1"], u_t, h1, c1)
    h2, c2 = lstm_cell(params["lstm2"], h1, h2, c2)
    m = jax.nn.sigmoid(h2 @ params["out"]["W"] + params["out"]["b"])
    return (h1, c1, h2, c2), m


###########################################################
#                         MODES                           #
###########################################################

def run_stream(params: dict, u: jnp.ndarray) -> jnp.ndarray:
    """u (B, T, d) -> masks (B, T, 64); memory carried across the whole utterance (causal)."""
    carry = zero_carry(params, u.shape[:1])
    _, m = jax.lax.scan(lambda c, x: step(params, c, x), carry, jnp.swapaxes(u, 0, 1))
    return jnp.swapaxes(m, 0, 1)


def run_window3(params: dict, u: jnp.ndarray, k: int = CFG.audio.lstm_context_frames) -> jnp.ndarray:
    """Keshavarzi's scheme: mask at frame j from frames j-2, j-1, j only, starting from a zero memory."""
    B, T, d = u.shape
    up = jnp.pad(u, ((0, 0), (k - 1, 0), (0, 0)))                         # zeros before the first frame
    win = jnp.stack([up[:, i: i + T] for i in range(k)], axis=2)           # (B, T, k, d)
    win = win.reshape(B * T, k, d)
    carry = zero_carry(params, (B * T,))
    _, m = jax.lax.scan(lambda c, x: step(params, c, x), carry, jnp.swapaxes(win, 0, 1))
    return m[-1].reshape(B, T, -1)


def apply(params: dict, u: jnp.ndarray, mode: str = "stream") -> jnp.ndarray:
    return {"stream": run_stream, "window3": run_window3}[mode](params, u)


def loss_fn(params: dict, u: jnp.ndarray, irm: jnp.ndarray, mode: str) -> jnp.ndarray:
    """Keshavarzi objective: mean squared error between predicted mask and ideal ratio mask."""
    return jnp.mean((apply(params, u, mode) - irm) ** 2)


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
