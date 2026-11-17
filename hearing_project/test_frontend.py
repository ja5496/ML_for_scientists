# Hearing project: tests for the gammatone front end - Jake Abraham
"""Run with:  cd hearing_project && python -m pytest -q test_frontend.py"""

import numpy as np
import jax.numpy as jnp

import frontend_gammatone as fe
from config import CFG

rng = np.random.default_rng(0)


def test_perfect_reconstruction():
    x = jnp.asarray(rng.standard_normal(16_000).astype(np.float32))
    y = fe.istft(fe.stft(x), x.shape[-1])
    assert y.shape == x.shape
    assert float(jnp.max(jnp.abs(y - x))) < 1e-4


def test_unity_mask_is_identity():
    x = jnp.asarray(rng.standard_normal(8_123).astype(np.float32))
    X = fe.stft(x)
    M = jnp.ones(X.shape[:-1] + (CFG.audio.n_bands,))
    y = fe.istft(fe.apply_mask(X, M), x.shape[-1])
    assert float(jnp.max(jnp.abs(y - x))) < 1e-4


def test_shadow_filtering_is_linear():
    s, v = (jnp.asarray(rng.standard_normal(4_000).astype(np.float32)) for _ in range(2))
    M = jnp.asarray(rng.uniform(size=(fe.n_frames(4_000), CFG.audio.n_bands)).astype(np.float32))
    y_mix = fe.enhance_with_mask(s + v, M)
    y_sum = fe.enhance_with_mask(s, M) + fe.enhance_with_mask(v, M)
    assert float(jnp.max(jnp.abs(y_mix - y_sum))) < 1e-4


def test_stft_is_causal():
    x = rng.standard_normal(4_000).astype(np.float32)
    X1 = fe.stft(jnp.asarray(x))
    t0 = 2_000
    x2 = x.copy(); x2[t0:] = rng.standard_normal(4_000 - t0)
    X2 = fe.stft(jnp.asarray(x2))
    # frame j covers samples up to 40 j + 39  ->  frames j <= (t0 - 40) / 40 must be unchanged
    last_safe = (t0 - 40) // 40
    assert np.allclose(X1[: last_safe + 1], X2[: last_safe + 1], atol=1e-5)
    assert not np.allclose(X1[last_safe + 1], X2[last_safe + 1])


def test_irm_bounds_and_batch_dims():
    s = jnp.asarray(rng.standard_normal((3, 2_000)).astype(np.float32))
    v = 3 * jnp.asarray(rng.standard_normal((3, 2_000)).astype(np.float32))
    M = fe.ideal_ratio_mask(s, v)
    assert M.shape == (3, fe.n_frames(2_000), CFG.audio.n_bands)
    assert float(M.min()) >= 0 and float(M.max()) <= 1


def test_filterbank_shape_and_coverage():
    assert fe.G_ANALYSIS.shape == (CFG.audio.n_bands, CFG.audio.n_fft // 2 + 1)
    assert np.allclose(fe.G_SYNTH.sum(axis=0), 1.0, atol=1e-5)
    fc = fe.erb_space()
    assert abs(fc[0] - 50) < 1e-6 and abs(fc[-1] - 8000) < 1e-3
