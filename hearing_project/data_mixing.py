# Hearing project: building transient-corrupted speech - Jake Abraham
"""
PLAIN WORDS
-----------
Makes the training examples: take a clean sentence, drop one transient (e.g. a crinkle) at a random
moment, and set how loud the transient is relative to the speech. That loudness ratio is the
"speech-to-transient ratio" (STR). Following Keshavarzi et al., the transient's loudness is measured
only in a 5-ms window around its peak — so −15 dB STR means the *peak* of the crinkle is much louder
than the average speech, which is what makes these sounds so unpleasant.

We keep the clean speech s and the scaled transient v separately as well as the mixture x = s + v.
That lets us (1) compute the ideal mask that the network should learn, and (2) later measure exactly
how much transient was removed and how much speech was damaged.

Mixtures are created fresh every time ("dynamic mixing"), so the network never sees the same
combination twice. Test mixtures are created once with a fixed seed so every model is scored on
exactly the same audio.

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
str_scale(s, v, target_str_db):
    peak = argmax |v|
    w = CFG.data.str_peak_window_ms * fs / 1000 samples, centered on peak
    rms_v = sqrt(mean(v[peak - w/2 : peak + w/2] ** 2))
    rms_s = sqrt(mean(s ** 2))                      # whole utterance, silences trimmed already
    gain = (rms_s / rms_v) * 10 ** (-target_str_db / 20)
    return gain * v

make_mixture(key, speech, transient, str_db, background=None, snr_db=None):
    onset = uniform(0.1 * len(speech), 0.9 * len(speech))    # one transient per sentence, inside the speech
    v = zeros_like(speech); paste transient starting at onset (truncate at end)
    v = str_scale(speech, v, str_db)
    n = 0
    if background is not None:  n = scale background to snr_db relative to speech
    x = speech + v + n
    # peak-normalise jointly to avoid clipping; apply the same gain to s, v, n
    return dict(x=x, s=speech, v=v, n=n, onset=onset, str_db=str_db)

IMPORTANT for the IRM with background noise: the "interference" is v + n, and the target is s.
(For the comfort analysis we still keep v and n separate.)

batch_iterator(split, key, batch_size, crop_frames):
    loop forever:
        sample batch_size utterances + transients + STRs from the split
        crop to crop_frames * hop samples (random crop that CONTAINS the transient onset)
        yield stacked arrays

build_fixed_test_set(seed):
    for each condition in {seen, unseen, grandma} x test_str_db x {quiet, babble 10 dB, babble 5 dB}:
        create N=300 mixtures (Keshavarzi used 300 sentences from 6 talkers)
    save as .npz with s, v, n, x and metadata; this file is the single source of truth for scoring.

Unit tests:
    - measured STR of a produced mixture equals the requested STR within 0.1 dB
    - x == s + v + n exactly
    - the same seed reproduces bit-identical test mixtures
"""

import jax
import jax.numpy as jnp

from config import CFG


def str_scale(s: jnp.ndarray, v: jnp.ndarray, target_str_db: float) -> jnp.ndarray:
    """Scale transient v so the speech-to-transient ratio (peak-window definition) equals target_str_db."""
    raise NotImplementedError


def make_mixture(key, speech, transient, str_db, background=None, snr_db=None) -> dict:
    """Return dict(x, s, v, n, onset, str_db). x = s + v + n."""
    raise NotImplementedError


def batch_iterator(split: str, key, batch_size: int = CFG.mask.batch_size, crop_frames: int = CFG.mask.seq_len_frames):
    """Infinite generator of dynamically mixed training batches."""
    raise NotImplementedError


def build_fixed_test_set(seed: int = 1234):
    """Create and save the frozen test mixtures used by every model."""
    raise NotImplementedError
