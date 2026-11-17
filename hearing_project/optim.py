# Hearing project: Adam + gradient clipping (same as Lab 2) - Jake Abraham
"""
PLAIN WORDS
-----------
The optimiser from Lab 2, unchanged in spirit: Adam with bias correction and global-norm gradient
clipping. Keshavarzi et al. used Adam with lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 and a
constant learning rate; that is the default here. A short warm-up and a cosine decay can be switched on.
"""

import jax
import jax.numpy as jnp

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def make_schedule(peak_lr: float, n_steps: int, warmup: int = 0, cosine: bool = False):
    def lr(t):
        w = jnp.minimum(1.0, t / max(warmup, 1)) if warmup else 1.0
        c = 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip(t / n_steps, 0, 1))) if cosine else 1.0
        return peak_lr * w * c
    return lr


def clip_by_global_norm(grads, clip: float):
    norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    return jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, clip / (norm + EPS)), grads), norm


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (zeros, zeros, jnp.array(0))


def adam_update(params, grads, state, lr_fn):
    m, v, t = state
    t = t + 1
    lr = lr_fn(t)
    m = jax.tree_util.tree_map(lambda m, g: BETA1 * m + (1 - BETA1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v, g: BETA2 * v + (1 - BETA2) * g ** 2, v, grads)
    mh, vh = 1 / (1 - BETA1 ** t), 1 / (1 - BETA2 ** t)
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m * mh) / (jnp.sqrt(v * vh) + EPS), params, m, v)
    return params, (m, v, t)


def n_params(params) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
