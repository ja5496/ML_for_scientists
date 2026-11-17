# Hearing project: tests for memories and attention - Jake Abraham
"""Run with:  cd hearing_project && python -m pytest -q test_attention.py"""

import jax
import jax.numpy as jnp
import numpy as np

import attention_feedback as af
import local_memory as lm

rng = np.random.default_rng(0)


def _toy_memories(U=12):
    ws = np.zeros((U, 2)); ws[:4, 0] = 1; ws[4:6, 1] = 1        # "speech" uses units 0-5
    wt = np.zeros((U, 1)); wt[8:12, 0] = 1                       # "transient" uses units 8-11
    n = lambda w: w / np.linalg.norm(w, axis=0, keepdims=True)
    return {"W_speech": n(ws), "W_trans": n(wt), "W_free": None}


def test_nmf_recovers_parts():
    W_true = np.abs(rng.normal(size=(10, 3))) * (rng.uniform(size=(10, 3)) > 0.5)
    A = W_true @ np.abs(rng.normal(size=(3, 2000)))
    W = lm.nmf(A, 3, n_iter=400)
    rel = np.linalg.norm(A - W @ np.linalg.lstsq(W, A, rcond=None)[0]) / np.linalg.norm(A)
    assert W.min() >= 0 and rel < 0.05


def test_beliefs_separate_toy_classes():
    mem = af.memory_matrix(_toy_memories())
    f_speech = np.zeros((1, 12, 5)); f_speech[0, :4] = 1.0
    f_trans = np.zeros((1, 12, 5)); f_trans[0, 8:12] = 1.0
    bs, bt = af.beliefs(jnp.asarray(f_speech), mem), af.beliefs(jnp.asarray(f_trans), mem)
    assert float(bs[0, 0].mean()) > 1.0 and float(bs[0, 1].mean()) < 0.1
    assert float(bt[0, 1].mean()) > 1.0 and float(bt[0, 0].mean()) < 0.1


def test_attention_is_causal_and_alpha_one_is_identity():
    mem = af.memory_matrix(_toy_memories())
    F = jnp.asarray(rng.uniform(size=(2, 12, 80)).astype(np.float32))
    a1 = af.apply(F, mem, hop_ms=2.5)
    a2 = af.apply(F.at[..., 40:].set(0.9), mem, hop_ms=2.5)
    for x, y in zip(a1, a2):
        assert jnp.allclose(x[..., :40], y[..., :40], atol=1e-5)
    F_att, _, _ = af.apply(F, mem, hop_ms=2.5, alpha=1.0)
    assert jnp.allclose(F_att, F)


def test_coherent_units_pass_incoherent_units_are_gated():
    T = 400
    t = np.arange(T)
    speech_env = (np.sin(2 * np.pi * t / 40) > 0).astype(np.float32)        # syllable-like on/off
    F = np.zeros((1, 12, T), np.float32)
    F[0, :4] = speech_env                                                      # speech units follow the envelope
    F[0, 8:12] = (rng.uniform(size=T) > 0.9)                                   # transient-ish units fire randomly
    F_att, F_rej, bel = af.apply(jnp.asarray(F), af.memory_matrix(_toy_memories()), hop_ms=2.5, beta=0.3)
    late = slice(200, None)
    kept_speech = float(F_att[0, :4, late].sum() / F[0, :4, late].sum())
    kept_trans = float(F_att[0, 8:12, late].sum() / max(F[0, 8:12, late].sum(), 1))
    assert kept_speech > kept_trans
    assert float(F_rej[0, 8:12, late].sum()) > float(F_rej[0, :4, late].sum())
