# Hearing project: non-neural comparison (MCTR) - Jake Abraham
"""
PLAIN WORDS
-----------
The "old-school" transient reducer Keshavarzi et al. compared against (variant B). It has no
learning: it splits sound into 5 frequency channels using tiny 1-ms frames, keeps a running average
of how loud each channel normally is, and whenever a channel suddenly jumps far above its average
it turns that channel down for that instant. It's a useful floor: any learned model should beat it.

IMPLEMENTATION NOTES (status: IMPLEMENTED; Keshavarzi Eqs. 3–4, "medium" setting alpha = 0.467)
------------------------------------------------------------------------------------------------
Steps: 16 kHz -> 22.05 kHz; 22-sample (1 ms) Tukey frames with 12-sample overlap (hop 10); 32-point FFT;
bins grouped into 5 channels; detect M/RMS > delta_i; attenuate by alpha * 20log10(M/RMS) dB; resynthesise
(weighted overlap-add); back to 16 kHz.
Choices the paper does not specify (stated here so they can be reported):
  - bin -> channel map for the 17 rfft bins: {0,1}, {2}, {3,4}, {5,6,7}, {8..16} (sizes 1,1,2,3,9 + DC and Nyquist)
  - running RMS: one-pole smoother of M^2 with a 50-ms time constant, using only PAST frames for the
    reference (so a transient cannot raise its own threshold); it is then updated with the current frame.
  - Tukey shape parameter 0.5.
`mctr` returns the per-frame channel gains so the same processing can be re-applied to the clean speech
and the transient separately (shadow filtering, see evaluate_metrics.py).
"""

import numpy as np
from scipy.signal import resample_poly
from scipy.signal.windows import tukey

DELTA = np.array([12.0, 21.0, 12.0, 8.0, 7.0])
ALPHA = 0.467
CHANNEL_BINS = [[0, 1], [2], [3, 4], [5, 6, 7], list(range(8, 17))]
FS_IN, FS_MCTR = 16_000, 22_050
FRAME, HOP, NFFT = 22, 10, 32
WIN = tukey(FRAME, 0.5)


def _frames(x):
    T = (len(x) - FRAME) // HOP + 1
    idx = np.arange(T)[:, None] * HOP + np.arange(FRAME)[None, :]
    return x[idx]


def _to_mctr_rate(x):
    y = resample_poly(np.asarray(x, np.float64), 441, 320)
    return np.pad(y, (FRAME, FRAME))


def _from_mctr_rate(y, n):
    return resample_poly(y[FRAME:-FRAME], 320, 441)[:n].astype(np.float32)


def detect_gains(x: np.ndarray, rms_tau_ms: float = 50.0, alpha: float = ALPHA) -> np.ndarray:
    """Per-frame, per-channel linear gains (T, 5) computed from the mixture x (16 kHz)."""
    X = np.fft.rfft(_frames(_to_mctr_rate(x)) * WIN, n=NFFT, axis=-1)
    P = np.abs(X) ** 2
    M = np.sqrt(np.stack([P[:, b].sum(axis=1) for b in CHANNEL_BINS], axis=1))     # (T, 5)
    lam = np.exp(-1.0 / (rms_tau_ms / 1000 * FS_MCTR / HOP))
    ms = M[0] ** 2 + 1e-12
    gains = np.ones_like(M)
    for j in range(M.shape[0]):
        ratio = M[j] / np.sqrt(ms)
        R = 20 * np.log10(ratio + 1e-12)
        hit = (ratio > DELTA) & (R > 0)
        gains[j] = np.where(hit, 10 ** (-alpha * R / 20), 1.0)
        ms = lam * ms + (1 - lam) * (M[j] ** 2 + 1e-12)
    return gains


def apply_gains(signal: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Apply per-frame channel gains to any 16-kHz signal of the same length as the one analysed."""
    xr = _to_mctr_rate(signal)
    X = np.fft.rfft(_frames(xr) * WIN, n=NFFT, axis=-1)
    g_bins = np.ones((gains.shape[0], NFFT // 2 + 1))
    for c, b in enumerate(CHANNEL_BINS):
        g_bins[:, b] = gains[:, [c]]
    frames = np.fft.irfft(X * g_bins, n=NFFT, axis=-1)[:, :FRAME] * WIN
    y, norm = np.zeros(len(xr)), np.zeros(len(xr))
    for j in range(frames.shape[0]):
        y[j * HOP: j * HOP + FRAME] += frames[j]
        norm[j * HOP: j * HOP + FRAME] += WIN ** 2
    return _from_mctr_rate(y / np.maximum(norm, 1e-8), len(signal))


def mctr(x: np.ndarray, rms_tau_ms: float = 50.0) -> tuple:
    """Return (processed_audio, gains). Re-use `gains` with apply_gains for shadow filtering."""
    gains = detect_gains(x, rms_tau_ms)
    return apply_gains(x, gains), gains
