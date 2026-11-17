# Hearing project: tests that nothing peeks into the future - Jake Abraham
"""
PLAIN WORDS
-----------
A hearing aid cannot know what sound is coming next. If any part of the model accidentally uses
future audio (very easy to do with slow, pooled layers), it will look better in testing than it
could ever be in a real device. These tests change the *future* part of an input and check that the
*past* outputs don't move. Run them after implementing each module — before trusting any result.

Run with:  cd hearing_project && python -m pytest -q test_causality.py

IMPLEMENTED: front end, mask estimator (both modes), full baseline pipelines C and D end to end.
TO ADD in stage 2–4 (same pattern): dbn_crbm.hidden_preact / prob_max_pool (pool 1 and 3),
dbn_distributed.hold_to_base for rates 1, 3, 9, 27 (+ exact check that slow frame k first appears at base
frame (k+1)R - 1), dbn_distributed.forward, attention_feedback.apply, variant H end to end,
enhance_stream.check_equivalence (streaming output == offline output).
"""

import jax
import jax.numpy as jnp
import numpy as np

import frontend_gammatone as fe
import mask_estimator_lstm as ml
import model_variants as mv
from config import CFG

rng = np.random.default_rng(0)
HOP = CFG.audio.hop_len


def last_safe_frame(t0: int) -> int:
    """Frame j covers samples up to 40 j + 39, so frames j <= (t0 - 40) / 40 never see sample t0."""
    return (t0 - HOP) // HOP


def _perturb_after(x, t0):
    x2 = x.copy()
    x2[..., t0:] = rng.standard_normal(x2[..., t0:].shape)
    return x2


def test_mask_net_causal_both_modes():
    p = ml.init_mask_net(jax.random.PRNGKey(0), CFG.audio.n_bands)
    u = jax.random.normal(jax.random.PRNGKey(1), (2, 300, CFG.audio.n_bands))
    for mode in ("stream", "window3"):
        a = ml.apply(p, u, mode)
        b = ml.apply(p, u.at[:, 150:].set(4.0), mode)
        assert jnp.allclose(a[:, :150], b[:, :150], atol=1e-6), mode


def test_full_baseline_pipeline_causal():
    stats = {"mean": jnp.zeros(CFG.audio.n_bands), "std": jnp.ones(CFG.audio.n_bands)}
    params = ml.init_mask_net(jax.random.PRNGKey(2), CFG.audio.n_bands)
    x = (0.05 * rng.standard_normal((1, 16_000))).astype(np.float32)
    t0 = 8_000
    x2 = _perturb_after(x, t0)
    for v in ("C", "D"):
        model = mv.Trained(v, params, stats)
        m1 = mv.neural_masks(model, [x[0]])[0]
        m2 = mv.neural_masks(model, [x2[0]])[0]
        k = last_safe_frame(t0)
        assert np.allclose(m1[: k + 1], m2[: k + 1], atol=1e-5), v
        assert not np.allclose(m1[k + 1:], m2[k + 1:]), v


def test_output_audio_latency_is_one_frame():
    """Output sample n may depend on input up to sample n + 79 (one 5-ms frame): the algorithmic delay."""
    M = jnp.asarray(rng.uniform(size=(fe.n_frames(8_000), CFG.audio.n_bands)).astype(np.float32))
    x = (0.05 * rng.standard_normal(8_000)).astype(np.float32)
    y1 = np.asarray(fe.enhance_with_mask(jnp.asarray(x), M))
    t0 = 4_000
    y2 = np.asarray(fe.enhance_with_mask(jnp.asarray(_perturb_after(x, t0)), M))
    delay = CFG.audio.frame_len - 1
    assert np.allclose(y1[: t0 - delay - 1], y2[: t0 - delay - 1], atol=1e-6)
