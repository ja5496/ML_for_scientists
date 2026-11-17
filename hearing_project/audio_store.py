# Hearing project: compact on-disk storage for many short audio clips - Jake Abraham
"""
PLAIN WORDS
-----------
Thousands of tiny audio files are slow to load. After downloading, every clip a split needs is
resampled to 16 kHz and packed end-to-end into one array saved as a single file, with a table that
says where each clip starts and ends plus its label/talker. Loading a whole split then takes a
fraction of a second, and the full first-pass dataset (~2,000 sentences + transients) fits in RAM.

A pack also records whether it is SYNTHETIC (made by synthetic_audio.py for smoke tests). Anything
produced from a synthetic pack is labelled as such and must never be reported as a result.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


@dataclass
class Pack:
    audio: np.ndarray        # float32, all clips concatenated
    offsets: np.ndarray      # int64, (n_clips + 1,), clip i = audio[offsets[i]:offsets[i+1]]
    meta: list               # one dict per clip (talker / label / source file / license ...)
    synthetic: bool = False

    def __len__(self):
        return len(self.meta)

    def clip(self, i: int) -> np.ndarray:
        return self.audio[self.offsets[i]: self.offsets[i + 1]]

    def field(self, key: str) -> list:
        return [m.get(key) for m in self.meta]


def save_pack(path: Path, clips: list, meta: list, synthetic: bool = False):
    assert len(clips) == len(meta) and len(clips) > 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(c) for c in clips], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    audio = np.concatenate([np.asarray(c, np.float32) for c in clips])
    np.savez(path, audio=audio, offsets=offsets, meta=json.dumps(meta), synthetic=synthetic)


def load_pack(path: Path) -> Pack:
    with np.load(Path(path), allow_pickle=False) as z:
        return Pack(audio=z["audio"], offsets=z["offsets"], meta=json.loads(str(z["meta"])),
                    synthetic=bool(z["synthetic"]))


def to_mono_16k(x: np.ndarray, fs: int) -> np.ndarray:
    """Mix to mono and resample to 16 kHz with a polyphase filter."""
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if fs != 16_000:
        g = np.gcd(int(fs), 16_000)
        x = resample_poly(x, 16_000 // g, int(fs) // g)
    return x.astype(np.float32)


def frame_levels_db(x: np.ndarray, fs: int = 16_000, win_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(fs * win_ms / 1000))
    T = len(x) // n
    if T == 0:
        return np.array([-200.0])
    e = (x[: T * n].reshape(T, n) ** 2).mean(axis=1)
    return 10 * np.log10(e + 1e-12)


def trim_silence(x: np.ndarray, fs: int = 16_000, below_peak_db: float = 40.0, pad_ms: float = 20.0) -> np.ndarray:
    """Keep from the first to the last 10-ms frame within `below_peak_db` of the loudest frame."""
    lv = frame_levels_db(x, fs)
    active = np.where(lv > lv.max() - below_peak_db)[0]
    n = int(fs * 0.01)
    pad = int(fs * pad_ms / 1000)
    start = max(0, active[0] * n - pad)
    end = min(len(x), (active[-1] + 1) * n + pad)
    return x[start:end]
