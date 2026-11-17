# Hearing project: the distributed belief network (tree of 15 sub-networks) - Jake Abraham
"""
PLAIN WORDS
-----------
This is the "auditory cortex" stage, Bellur & Elhilali's Fig. 1. Instead of one deep stack, it is a
tree: one sub-network sits on the 64-band "ear" output, and each sub-network feeds two children —
a left child that keeps the same time resolution and a right child that runs 3x slower. Four layers
give 1 + 2 + 4 + 8 = 15 sub-networks ("LSNs"). The rightmost path ends up looking at ~0.45 s of the
past with coarse timing; the leftmost stays fast and detailed. Together they describe the same sound
from 15 "vantage points" — redundant on purpose, because a crinkle that fools one view usually
doesn't fool all of them.

Between layers, the attention gate (attention_feedback.py) can nudge each sub-network toward
speech-like activity before its output is passed upward — exactly where Bellur put it.

Because the sub-networks run at different speeds, their outputs are "held" so that every one of them
has a value at every 2.5-ms frame, without ever peeking at the future. This is the most
bug-prone part of the project and has its own test (test_causality.py).

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
Indexing (Bellur's notation): LSN (g, z), layer g = 1..4, position z = 1..2^(g-1).
    parent(g, z) = (g-1, ceil(z/2))
    z odd  -> left child  -> pool = CFG.dbn.pool_left  (1)
    z even -> right child -> pool = CFG.dbn.pool_right (3)
    root (1,1): pool = 1, input = 64 log-gammatone bands (Gaussian visible, z-scored)
    others: input = parent's (attended) output, U channels, Bernoulli visible
    rate[g, z] = product of pools along the path from root = 3 ** (# right turns)  (in 2.5-ms base frames)
    e.g. LSN48 (rightmost, layer 4): rate 27 -> one output every 67.5 ms, receptive field ~0.45 s
    (see receptive_field_frames(); for Bellur-like ~1 s use pool_right=4 or a 5-ms root pool)

init_dbn(key) -> dict {(g, z): crbm_params}

forward(params, feats_fb, attn=None, memories=None) -> dict:
    # feats_fb: (64, T) log-gammatone, z-scored. Batch with vmap over utterances.
    outputs = {}
    for g in 1..4, for z in 1..2^(g-1):                     # parents before children
        v = feats_fb if g == 1 else outputs[parent(g, z)]["F_att_native"]
        F = crbm.encode(params[(g, z)], v, pool[(g, z)])    # (U, T / rate) at its NATIVE rate
        if attn is not None:
            F_att, F_rej, beliefs = attn.apply(F, memories[(g, z)], hop_ms = 2.5 * rate[(g, z)])
        else:
            F_att, F_rej, beliefs = F, zeros_like(F), None
        outputs[(g, z)] = dict(F_att_native = F_att,
                               F_att = hold_to_base(F_att, rate, T),
                               F_rej = hold_to_base(F_rej, rate, T),
                               beliefs = hold_to_base(beliefs, rate, T) if beliefs else None)
    return outputs     # 15 entries; every *_base array is (U, T) aligned to the 2.5-ms frames

hold_to_base(y_native (U, T_slow), rate R, T) -> (U, T):      # CAUSAL zero-order hold
    slow frame k summarises base frames up to (k+1)*R - 1  -> usable from base frame (k+1)*R - 1 on
    for base frame t:  k = floor((t + 1) / R) - 1
                       out[:, t] = y_native[:, k] if k >= 0 else 0
    Implement with jnp.repeat + a shift of (R - 1) frames, then crop to T. Double-check the shift
    with test_causality.py — getting this wrong by one frame leaks the future.

Streaming version (for enhance_stream.py / deployment):
    state per LSN = ring buffer of its last W input frames + pooling accumulator + attention EMA stats
    on each new base frame: update root; any LSN whose pooling block just completed emits a new
    frame and passes it to its children; held values persist otherwise.
    Must produce outputs IDENTICAL (within 1e-5) to forward() on the same audio.

Cost accounting (report in the paper/poster):
    MACs per base frame = sum over LSNs of (U * C_in * W) / rate
    With U=48, W=12: root ~37k MACs/frame, whole tree ~O(100k) MACs/frame (~40 MMAC/s at 400 frames/s).
"""

import jax
import jax.numpy as jnp

from config import CFG


def lsn_ids(n_layers: int = CFG.dbn.n_layers) -> list:
    """[(1,1), (2,1), (2,2), (3,1), ..., (4,8)] in parent-before-child order."""
    return [(g, z) for g in range(1, n_layers + 1) for z in range(1, 2 ** (g - 1) + 1)]


def parent(g: int, z: int) -> tuple:
    return (g - 1, (z + 1) // 2)


def pool_of(g: int, z: int) -> int:
    if g == 1:
        return 1
    return CFG.dbn.pool_left if z % 2 == 1 else CFG.dbn.pool_right


def rate_of(g: int, z: int) -> int:
    """Total downsampling of LSN (g, z) relative to the 2.5-ms base frame rate."""
    r = 1
    while g > 1:
        r *= pool_of(g, z)
        g, z = parent(g, z)
    return r


def receptive_field_frames(g: int, z: int, width: int = CFG.dbn.basis_width) -> int:
    """How many past base frames (2.5 ms each) can influence one output of LSN (g, z)."""
    if g == 1:
        return width
    pg, pz = parent(g, z)
    r_in = rate_of(pg, pz)                      # the child's input arrives at the parent's rate
    return receptive_field_frames(pg, pz, width) + (width - 1) * r_in + (pool_of(g, z) - 1) * r_in


def init_dbn(key) -> dict:
    raise NotImplementedError


def hold_to_base(y_native: jnp.ndarray, rate: int, T: int) -> jnp.ndarray:
    """Causal zero-order hold from an LSN's native rate to the base frame rate."""
    raise NotImplementedError


def forward(params: dict, feats_fb: jnp.ndarray, attn=None, memories=None) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    # Quick sanity printout of the tree (this part runs without any implementation).
    for g, z in lsn_ids():
        r = rate_of(g, z)
        print(f"LSN{g}{z}: parent={parent(g, z) if g > 1 else 'filterbank'}  pool={pool_of(g, z)}  "
              f"hop={2.5 * r:6.1f} ms  receptive field={2.5 * receptive_field_frames(g, z):6.1f} ms")
