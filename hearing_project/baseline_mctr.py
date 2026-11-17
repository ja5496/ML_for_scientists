# Hearing project: non-neural comparison (MCTR) - Jake Abraham
"""
PLAIN WORDS
-----------
The "old-school" transient reducer Keshavarzi et al. compared against (variant B). It has no
learning: it splits sound into 5 frequency channels using tiny 1-ms frames, keeps a running average
of how loud each channel normally is, and whenever a channel suddenly jumps far above its average
it turns that channel down for that instant. It's a useful floor: any learned model should beat it.

PSEUDOCODE FOR THE IMPLEMENTING AGENT  (Keshavarzi Eqs. 3–4, "medium" setting)
-------------------------------------
1. resample x: 16 kHz -> 22.05 kHz
2. frames of 22 samples (1 ms), overlap 12 samples, Tukey window
3. 32-point FFT -> 16 bins; group bins into 5 channels with sizes [1, 1, 2, 3, 9]
4. M[i, j] = magnitude of channel i in frame j
   RMS[i, j] = running RMS estimate (one-pole smoother; paper does not give the time constant —
               use 50 ms and report it as an assumption)
5. R[i, j] = 20 log10(M[i, j] / RMS[i, j])
   transient if M/RMS > delta_i, with delta = [12, 21, 12, 8, 7]
6. attenuation in dB: C = alpha * R if transient else 0, with alpha = 0.467
   apply gain 10 ** (-C / 20) to that channel's bins in that frame
7. inverse FFT + overlap-add, resample back to 16 kHz

The mask-equivalent gain (per frame, per band) should also be exported so shadow filtering
(apply the same gains to s and v separately) works for this method too.
"""

import jax.numpy as jnp

DELTA = (12.0, 21.0, 12.0, 8.0, 7.0)
ALPHA = 0.467
BINS_PER_CHANNEL = (1, 1, 2, 3, 9)


def mctr(x: jnp.ndarray, fs: int = 16_000, rms_tau_ms: float = 50.0) -> tuple:
    """Return (processed_audio, gains) where gains can be re-applied to s and v for shadow filtering."""
    raise NotImplementedError
