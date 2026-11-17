# Hearing project: the "ear" — gammatone analysis and resynthesis - Jake Abraham
"""
PLAIN WORDS
-----------
This is the cochlea stage, kept from Keshavarzi et al. The sound is cut into 5-ms slices (each
overlapping the previous one by half), each slice is turned into a spectrum, and the spectrum is
grouped into 64 bands spaced like the human ear (narrow bands at low frequencies, wide at high).
The energy in each band, on a log scale, is what the networks "hear".

It also does the reverse: given a gain between 0 and 1 for each of the 64 bands in each slice (the
mask), it turns the gains back into per-frequency gains, multiplies the original spectrum, and
stitches the slices back together into audio (overlap-add). Because the mask is applied in the same
domain the ear stage uses, the belief network never has to produce audio itself — it only supplies
evidence to the mask estimator.

Important: this stage is *shared*. The baseline, the proposed hybrid, and the DBN all start from
these same 64-band energies, so any improvement can't be credited to a better front end.

A side effect of 5-ms slices: the spectrum only has a point every 125 Hz, so the narrowest low-
frequency bands (below ~300 Hz) see nearly the same few points and carry redundant information.
Keshavarzi et al. accepted this to keep the delay short; so do we.

IMPLEMENTATION NOTES (status: IMPLEMENTED, tested in test_frontend.py)
----------------------------------------------------------------------
- All functions take arrays with arbitrary leading batch dims: x (..., N) -> X (..., T, 65).
- Window: periodic sqrt-Hann for analysis AND synthesis (product = Hann, which overlap-adds to 1 at 50%
  overlap -> perfect reconstruction). Keshavarzi used a plain Hann analysis window; the difference only
  affects feature scaling and is removed by z-scoring.
- Framing: x is left-padded by (frame_len - hop) = 40 zeros. Frame j covers original samples
  [40j - 40, 40j + 39]. Frame j therefore never depends on samples after 40j + 39 (causal).
- Gammatone weights: 4th-order gammatone POWER response |H(f)|^2 = (1 + ((f - fc)/b)^2)^-4,
  b = 1.019 ERB(fc), centre frequencies equally spaced on the ERB-number scale, 50 Hz – 8 kHz.
- Synthesis weights are column-normalised so a mask of all ones returns the input exactly.
- IRM = (S2 / (S2 + V2)) ** beta in the band-energy domain (beta = 0.5).
"""

import math

import jax.numpy as jnp
import numpy as np

from config import CFG

A = CFG.audio
assert A.frame_len == 2 * A.hop_len, "overlap-add below assumes exactly 50% overlap"
N_BINS = A.n_fft // 2 + 1


###########################################################
#                   FILTERBANK WEIGHTS                    #
###########################################################

def _erb_num(f):
    return 21.4 * np.log10(4.37e-3 * f + 1.0)


def _erb_num_inv(e):
    return (10 ** (e / 21.4) - 1.0) / 4.37e-3


def erb_space(fmin=A.f_min_hz, fmax=A.f_max_hz, n=A.n_bands) -> np.ndarray:
    """Centre frequencies (Hz) equally spaced on the ERB-number scale (Glasberg & Moore 1990)."""
    return _erb_num_inv(np.linspace(_erb_num(fmin), _erb_num(fmax), n))


def gammatone_weights() -> tuple:
    """Return (G_analysis, G_synth), each (n_bands, n_fft//2 + 1).

    G_analysis[b, k] = gammatone power response of band b at FFT bin k (peak 1).
    G_synth = G_analysis normalised over bands so that sum_b G_synth[b, k] = 1 for every bin k.
    """
    fc = erb_space()
    f = np.arange(N_BINS) * A.sample_rate / A.n_fft
    erb = 24.7 * (4.37e-3 * fc + 1.0)
    bw = 1.019 * erb
    G = (1.0 + ((f[None, :] - fc[:, None]) / bw[:, None]) ** 2) ** -4
    G_synth = G / G.sum(axis=0, keepdims=True)
    return jnp.asarray(G, jnp.float32), jnp.asarray(G_synth, jnp.float32)


G_ANALYSIS, G_SYNTH = gammatone_weights()


###########################################################
#                     STFT / ISTFT                        #
###########################################################

def _sqrt_hann(n):
    return jnp.sqrt(0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n) / n)).astype(jnp.float32)


WINDOW = _sqrt_hann(A.frame_len)


def n_frames(n_samples: int) -> int:
    """Number of frames produced for a signal of n_samples."""
    padded = A.frame_len - A.hop_len + n_samples
    padded = int(math.ceil(padded / A.hop_len)) * A.hop_len + A.hop_len
    return (padded - A.frame_len) // A.hop_len + 1


def frame_signal(x: jnp.ndarray) -> jnp.ndarray:
    """x (..., N) -> frames (..., T, frame_len). Causal: left pad only affects the start."""
    N = x.shape[-1]
    T = n_frames(N)
    total = (T - 1) * A.hop_len + A.frame_len
    left = A.frame_len - A.hop_len
    pad = [(0, 0)] * (x.ndim - 1) + [(left, total - left - N)]
    xp = jnp.pad(x, pad)
    idx = jnp.arange(T)[:, None] * A.hop_len + jnp.arange(A.frame_len)[None, :]
    return xp[..., idx]


def stft(x: jnp.ndarray) -> jnp.ndarray:
    """x (..., N) real -> X (..., T, n_fft//2 + 1) complex."""
    return jnp.fft.rfft(frame_signal(x) * WINDOW, n=A.n_fft, axis=-1)


def istft(Y: jnp.ndarray, length: int) -> jnp.ndarray:
    """Inverse of stft by windowed overlap-add. Y (..., T, bins) -> y (..., length)."""
    frames = jnp.fft.irfft(Y, n=A.n_fft, axis=-1)[..., : A.frame_len] * WINDOW
    h = A.hop_len
    first, second = frames[..., :h], frames[..., h:]                 # (..., T, hop)
    zeros = jnp.zeros_like(first[..., :1, :])
    blocks = jnp.concatenate([first, zeros], axis=-2) + jnp.concatenate([zeros, second], axis=-2)
    y = blocks.reshape(*blocks.shape[:-2], -1)
    left = A.frame_len - A.hop_len
    return y[..., left: left + length]


###########################################################
#                 BAND ENERGIES / MASKS                   #
###########################################################

def band_energies(X: jnp.ndarray, G: jnp.ndarray = G_ANALYSIS) -> jnp.ndarray:
    """X (..., T, bins) complex -> E (..., T, n_bands) band energies."""
    return (jnp.abs(X) ** 2) @ G.T


def log_features(E: jnp.ndarray, stats: dict = None) -> jnp.ndarray:
    """log-compress band energies; z-score with TRAINING-set stats if given ({'mean','std'} per band)."""
    L = jnp.log(E + A.log_floor)
    if stats is not None:
        L = (L - stats["mean"]) / stats["std"]
    return L


def ideal_ratio_mask(s: jnp.ndarray, interference: jnp.ndarray, beta: float = A.irm_beta) -> jnp.ndarray:
    """IRM in the band domain. s, interference: (..., N) waveforms. Returns (..., T, n_bands) in [0, 1]."""
    S2 = band_energies(stft(s))
    V2 = band_energies(stft(interference))
    return (S2 / (S2 + V2 + 1e-12)) ** beta


def apply_mask(X_mix: jnp.ndarray, M: jnp.ndarray, G_synth: jnp.ndarray = G_SYNTH) -> jnp.ndarray:
    """Band mask M (..., T, n_bands) -> per-bin real gain -> masked spectrum (mixture phase kept)."""
    return X_mix * (M @ G_synth)


def enhance_with_mask(x: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    """Convenience: waveform x + band mask M -> processed waveform of the same length."""
    return istft(apply_mask(stft(x), M), x.shape[-1])
