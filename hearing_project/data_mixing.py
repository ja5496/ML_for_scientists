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

Training mixtures are created fresh every time ("dynamic mixing"), so the network never sees the same
combination twice. The validation and test mixtures are fixed "recipes" (which sentence, which
transient, where, how loud) saved once with a fixed seed, so every model is scored on exactly the
same audio. Saving recipes instead of audio keeps them tiny.

IMPLEMENTATION NOTES (status: IMPLEMENTED, tested in test_data_mixing.py)
-------------------------------------------------------------------------
- Everything here is numpy (CPU); arrays are handed to JAX only inside the model/training code.
- Speech is normalised to RMS = CFG.data.speech_rms before the transient is scaled.
- Training crops are 2 s (CFG.mask.seq_len_frames * hop) and always contain the transient's peak.
- Background noise (stretch condition) needs LibriSpeech/DEMAND and is left for stage 2+.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_store import Pack, load_pack
from config import CFG

D = CFG.data
FS = CFG.audio.sample_rate
PRE_PEAK = int(FS * D.transient_pre_peak_ms / 1000)


###########################################################
#                         MIXING                          #
###########################################################

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def peak_window_rms(v: np.ndarray, window_ms: float = D.str_peak_window_ms) -> float:
    """RMS of v in a window centred on its absolute peak (Keshavarzi's STR definition)."""
    p = int(np.argmax(np.abs(v)))
    h = max(1, int(FS * window_ms / 1000) // 2)
    return rms(v[max(0, p - h): p + h])


def str_scale(s: np.ndarray, v: np.ndarray, target_str_db: float) -> np.ndarray:
    """Scale transient v so that 20 log10(rms(s) / peak-window rms(v)) == target_str_db."""
    return v * (rms(s) / peak_window_rms(v)) * 10 ** (-target_str_db / 20)


def measured_str_db(s: np.ndarray, v: np.ndarray) -> float:
    return 20 * np.log10(rms(s) / peak_window_rms(v))


def make_mixture(speech: np.ndarray, transient: np.ndarray, str_db: float, onset: int) -> dict:
    """One transient pasted into one sentence. Returns dict(x, s, v, n, onset, str_db); x = s + v + n."""
    s = speech * (D.speech_rms / rms(speech))
    v = np.zeros_like(s)
    seg = transient[: len(s) - onset]
    v[onset: onset + len(seg)] = seg
    v = str_scale(s, v, str_db)
    n = np.zeros_like(s)
    return dict(x=(s + v + n).astype(np.float32), s=s.astype(np.float32), v=v.astype(np.float32),
                n=n, onset=int(onset), str_db=float(str_db))


def random_onset(rng: np.random.Generator, n_speech: int) -> int:
    """Clip start such that the transient's PEAK lands between 10% and 90% of the sentence."""
    peak = rng.integers(int(0.1 * n_speech), int(0.9 * n_speech))
    return int(max(0, peak - PRE_PEAK))


###########################################################
#                          DATA                           #
###########################################################

@dataclass
class Corpus:
    speech: dict          # split -> Pack  ("train", "val", "test")
    transients: dict      # name  -> Pack  ("train", "val", "test_seen", "test_unseen", "test_grandma")
    synthetic: bool


def load_corpus(cache_dir: Path = D.cache_dir) -> Corpus:
    cache_dir = Path(cache_dir)
    speech = {sp: load_pack(cache_dir / f"speech_{sp}.npz") for sp in ("train", "val", "test")
              if (cache_dir / f"speech_{sp}.npz").exists()}
    trans = {p.stem[len("transient_"):]: load_pack(p) for p in sorted(cache_dir.glob("transient_*.npz"))}
    if not speech or not trans:
        raise FileNotFoundError(f"no packs in {cache_dir}: run data_download.py (or synthetic_audio.py for a smoke test)")
    synthetic = any(p.synthetic for p in list(speech.values()) + list(trans.values()))
    return Corpus(speech, trans, synthetic)


def batch_iterator(corpus: Corpus, split: str = "train", batch_size: int = CFG.mask.batch_size,
                   crop_frames: int = CFG.mask.seq_len_frames, seed: int = 0,
                   str_choices: tuple = D.train_str_db):
    """Infinite generator of dynamically mixed batches: dict of (B, L) float32 arrays + metadata."""
    rng = np.random.default_rng(seed)
    sp, tr = corpus.speech[split], corpus.transients[split]
    L = crop_frames * CFG.audio.hop_len
    while True:
        xs, ss, vs, onsets, strs = [], [], [], [], []
        for _ in range(batch_size):
            speech = sp.clip(rng.integers(len(sp)))
            trans = tr.clip(rng.integers(len(tr)))
            str_db = float(rng.choice(str_choices))
            m = make_mixture(speech, trans, str_db, random_onset(rng, len(speech)))
            peak = m["onset"] + PRE_PEAK
            if len(speech) <= L:
                start = 0
            else:
                lo = max(0, peak - L + int(0.3 * FS))
                hi = max(lo, min(peak - int(0.1 * FS), len(speech) - L))
                start = int(rng.integers(lo, hi + 1))
            def crop(a):
                out = np.zeros(L, np.float32)
                seg = a[start: start + L]
                out[: len(seg)] = seg
                return out
            xs.append(crop(m["x"])); ss.append(crop(m["s"])); vs.append(crop(m["v"]))
            onsets.append(m["onset"] - start); strs.append(str_db)
        yield dict(x=np.stack(xs), s=np.stack(ss), v=np.stack(vs),
                   onset=np.array(onsets), str_db=np.array(strs))


###########################################################
#                 FIXED VAL / TEST RECIPES                #
###########################################################

TEST_CONDITIONS = {"seen": "test_seen", "unseen": "test_unseen", "grandma": "test_grandma"}


def build_recipes(corpus: Corpus, seed: int = 1234) -> dict:
    """Fixed recipes. val: every val sentence x train STRs. test: every test sentence x condition x test STR."""
    rng = np.random.default_rng(seed)
    rows = []
    def add(set_name, cond, speech_split, trans_name, strs):
        sp, tr = corpus.speech[speech_split], corpus.transients[trans_name]
        for str_db in strs:
            for i in range(len(sp)):
                rows.append(dict(set=set_name, cond=cond, speech_split=speech_split, speech_idx=i,
                                 trans=trans_name, trans_idx=int(rng.integers(len(tr))),
                                 onset=random_onset(rng, len(sp.clip(i))), str_db=float(str_db)))
    add("val", "val", "val", "val", D.train_str_db)
    for cond, tname in TEST_CONDITIONS.items():
        if tname in corpus.transients:
            add("test", cond, "test", tname, D.test_str_db)
    return {"seed": seed, "synthetic": corpus.synthetic, "rows": rows}


def save_recipes(recipes: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipes))


def load_recipes(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def realise(corpus: Corpus, row: dict) -> dict:
    """Turn one recipe row back into audio (bit-identical every time)."""
    m = make_mixture(corpus.speech[row["speech_split"]].clip(row["speech_idx"]),
                     corpus.transients[row["trans"]].clip(row["trans_idx"]), row["str_db"], row["onset"])
    m.update({k: row[k] for k in ("set", "cond")})
    return m
