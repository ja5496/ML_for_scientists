# Hearing project: tests for the belief network - Jake Abraham
"""Run with:  cd hearing_project && python -m pytest -q test_dbn.py"""

import jax
import jax.numpy as jnp
import numpy as np

import dbn_crbm as cr
import dbn_distributed as dd
from config import CFG
from optim import adam_init, adam_update, make_schedule

U = CFG.dbn.units_per_lsn


def test_hidden_preact_is_causal():
    p = cr.init_crbm(jax.random.PRNGKey(0), 64)
    v = jax.random.normal(jax.random.PRNGKey(1), (2, 64, 100))
    a1 = cr.hidden_preact(p, v)
    a2 = cr.hidden_preact(p, v.at[..., 60:].set(3.0))
    assert a1.shape == (2, U, 100)
    assert jnp.allclose(a1[..., :60], a2[..., :60], atol=1e-5)
    assert not jnp.allclose(a1[..., 60], a2[..., 60])


def test_pooled_frames_are_causal():
    p = cr.init_crbm(jax.random.PRNGKey(0), 64)
    v = jax.random.normal(jax.random.PRNGKey(1), (1, 64, 99))
    for P in (1, 3):
        y1 = cr.encode(p, v, P)
        y2 = cr.encode(p, v.at[..., 60:].set(3.0), P)   # pooled frame m uses inputs <= mP + P - 1
        m_safe = (60 - P) // P                            # last pooled frame whose block ends before 60
        assert y1.shape[-1] == 99 // P
        assert jnp.allclose(y1[..., : m_safe + 1], y2[..., : m_safe + 1], atol=1e-6)


def test_pooling_values():
    """P = 1 must equal a plain sigmoid; P = 3 must equal 1 - prod-free formula S / (1 + S)."""
    p = cr.init_crbm(jax.random.PRNGKey(3), 64)
    v = jax.random.normal(jax.random.PRNGKey(4), (2, 64, 30))
    a = cr.hidden_preact(p, v)
    assert jnp.allclose(cr.encode(p, v, 1), jax.nn.sigmoid(a), atol=1e-6)
    S = jnp.exp(a).reshape(2, U, 10, 3).sum(-1)
    assert jnp.allclose(cr.encode(p, v, 3), S / (1 + S), atol=1e-6)
    det, pooled = cr.prob_max_pool(a, 3)
    assert jnp.allclose(det.reshape(2, U, 10, 3).sum(-1), pooled, atol=1e-6)   # P(block on) = sum of unit probs
    assert float(jax.nn.sigmoid(a).mean()) < 0.15                              # init bias -> ~5% firing


def test_hold_to_base_exact_timing():
    for R in (1, 3, 9, 27):
        Tn = 10
        y = jnp.arange(1, Tn + 1, dtype=jnp.float32)[None, :]       # slow frame k has value k + 1
        out = np.asarray(dd.hold_to_base(y, R, Tn * R))[0]
        for t in range(Tn * R):
            k = (t + 1) // R - 1
            assert out[t] == (k + 1 if k >= 0 else 0)                 # slow frame k first visible at t = (k+1)R - 1


def test_tree_forward_shapes_and_causality():
    params = dd.init_dbn(jax.random.PRNGKey(0))
    T = 270
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 64, T))
    out1 = dd.to_base_rate(dd.forward(params, x), T)
    out2 = dd.to_base_rate(dd.forward(params, x.at[..., 150:].set(2.0)), T)
    assert len(out1) == 15
    for lsn in dd.lsn_ids():
        assert out1[lsn].shape == (1, U, T)
        assert jnp.allclose(out1[lsn][..., :150], out2[lsn][..., :150], atol=1e-5), lsn


def test_cd_training_reduces_reconstruction_error():
    """Toy check: a CRBM trained by CD on structured inputs reconstructs them better than at init."""
    key = jax.random.PRNGKey(0)
    t = jnp.arange(120)
    patterns = jnp.stack([jnp.sin(2 * jnp.pi * (f * t / 120 + jnp.arange(16)[:, None] / 16)) for f in (2, 3, 5, 7)])
    p = cr.init_crbm(key, 16, n_units=24, width=8)
    opt = adam_init(p)
    lr = make_schedule(3e-3, 300)
    step = jax.jit(lambda p, o, v, k: (lambda lg: (lg, adam_update(p, lg[1], o, lr)))(
        jax.value_and_grad(lambda q: cr.cd_loss(q, v, k, "gaussian", 1), has_aux=True)(p)))
    first = None
    for i in range(300):
        v = patterns[jax.random.randint(jax.random.PRNGKey(i), (8,), 0, 4)]
        ((loss, m), g), (p, opt) = step(p, opt, v, jax.random.PRNGKey(1000 + i))
        first = first if first is not None else float(m["recon_mse"])
    assert float(m["recon_mse"]) < 0.5 * first
