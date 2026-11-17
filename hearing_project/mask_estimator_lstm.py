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

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
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


def init_lstm(key, d_in: int, d_hidden: int) -> dict:
    raise NotImplementedError


def lstm_cell(params: dict, x: jnp.ndarray, h: jnp.ndarray, c: jnp.ndarray) -> tuple:
    raise NotImplementedError


def init_mask_net(key, variant: str) -> dict:
    raise NotImplementedError


def step(params: dict, carry: dict, x_t: tuple) -> tuple:
    raise NotImplementedError


def apply(params: dict, features: dict, mode: str = "stream") -> jnp.ndarray:
    raise NotImplementedError


def loss_fn(params: dict, batch: dict, variant: str) -> jnp.ndarray:
    raise NotImplementedError


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
