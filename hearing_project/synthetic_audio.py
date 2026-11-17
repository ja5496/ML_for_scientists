# Hearing project: synthetic stand-in data for smoke tests ONLY - Jake Abraham
"""
PLAIN WORDS
-----------
Makes fake "speech" (buzzy vowel-like sounds with changing pitch and formants, plus hissy consonant
bursts) and fake "transients" (clicks, knocks, clanks, glass pings, crinkles) in exactly the same
format as the real data. Its only purpose is to check that every piece of code runs end to end in
a couple of minutes, without waiting for downloads.

Nothing trained or measured on this data means anything about real hearing-aid performance. Packs
made here carry a `synthetic` flag, and every figure or table produced from them is stamped
"SYNTHETIC SMOKE TEST".

USAGE:  python synthetic_audio.py      # writes packs to data/synthetic_cache/
"""

import numpy as np
from scipy.signal import butter, lfilter

from audio_store import save_pack
from config import CFG, DATA
from data_download import cut_transient

FS = CFG.audio.sample_rate
OUT = DATA / "synthetic_cache"
VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (570, 840, 2410), (300, 870, 2240)]


def _resonator(x, f, bw):
    r = np.exp(-np.pi * bw / FS)
    a = [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r]
    return lfilter([1 - r], a, x)


def fake_sentence(rng, f0_base):
    segs = []
    for _ in range(rng.integers(6, 14)):
        dur = int(FS * rng.uniform(0.12, 0.3))
        t = np.arange(dur) / FS
        env = np.sin(np.pi * np.arange(dur) / dur) ** 0.7
        if rng.uniform() < 0.75:                                  # voiced syllable
            f0 = f0_base * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * t + rng.uniform(0, 6)))
            phase = 2 * np.pi * np.cumsum(f0) / FS
            src = sum(np.sin(k * phase) / k for k in range(1, int(4000 / f0_base)))
            F = VOWELS[rng.integers(len(VOWELS))]
            y = sum(_resonator(src, f, 80 + 0.05 * f) for f in F)
        else:                                                     # fricative
            b, a = butter(4, rng.uniform(2500, 4500) / (FS / 2), "high")
            y = lfilter(b, a, rng.standard_normal(dur)) * 0.3
        segs.append(y * env)
        segs.append(np.zeros(int(FS * rng.uniform(0.0, 0.08))))
    x = np.concatenate(segs)
    return (x / (np.std(x) + 1e-9) * 0.05).astype(np.float32)


def fake_transient(rng, kind):
    n = int(0.6 * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    start = int(0.05 * FS)
    if kind == "click":
        x[start] = 1.0; x = _resonator(x, rng.uniform(2000, 5000), 300)
    elif kind == "knock":
        x[start:start + 40] = rng.standard_normal(40); x = _resonator(x, rng.uniform(150, 400), 60)
    elif kind == "clank":
        for f in rng.uniform(400, 3000, 5):
            x[start:] += np.sin(2 * np.pi * f * t[: n - start]) * np.exp(-t[: n - start] / rng.uniform(0.05, 0.2))
    elif kind == "glass":
        for f in rng.uniform(3000, 7000, 3):
            x[start:] += np.sin(2 * np.pi * f * t[: n - start]) * np.exp(-t[: n - start] / 0.15)
    elif kind == "crinkle":                                       # grandma's sound: dense random micro-clicks
        for p in start + np.sort(rng.integers(0, int(0.35 * FS), rng.integers(40, 120))):
            x[p] += rng.uniform(0.2, 1.0) * rng.choice([-1, 1])
        b, a = butter(2, 1500 / (FS / 2), "high"); x = lfilter(b, a, x)
    elif kind == "shake":
        for p in start + np.sort(rng.integers(0, int(0.2 * FS), 8)):
            x[p: p + 30] += rng.standard_normal(30)
        x = _resonator(x, rng.uniform(1000, 3000), 400)
    return x.astype(np.float32)


def build(seed=0, talkers=(16, 4, 6), utts=(10, 5, 5), per_class=24):
    rng = np.random.default_rng(seed)
    for split, n_t, k in zip(("train", "val", "test"), talkers, utts):
        clips, meta = [], []
        for t in range(n_t):
            f0 = rng.uniform(95, 230)
            for _ in range(k):
                clips.append(fake_sentence(rng, f0)); meta.append(dict(talker=f"{split}{t}", synthetic=True))
        save_pack(OUT / f"speech_{split}.npz", clips, meta, synthetic=True)
    def tclips(kinds, n):
        out = [(cut_transient(fake_transient(rng, k) + 1e-4 * rng.standard_normal(int(0.6 * FS)).astype(np.float32)), k)
               for k in kinds for _ in range(n)]
        return [(c, k) for c, k in out if c is not None]
    seen, unseen = ["click", "knock", "clank"], ["glass", "shake"]
    packs = {"train": tclips(seen, per_class), "val": tclips(seen, 4), "test_seen": tclips(seen, 4),
             "test_unseen": tclips(unseen, 6), "test_grandma": tclips(["crinkle"], 10)}
    for name, items in packs.items():
        save_pack(OUT / f"transient_{name}.npz", [c for c, _ in items], [dict(label=k, synthetic=True) for _, k in items],
                  synthetic=True)
    return OUT


if __name__ == "__main__":
    print("synthetic packs written to", build())
