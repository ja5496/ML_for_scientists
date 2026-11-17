# Hearing project: "local memories" of speech and transients - Jake Abraham
"""
PLAIN WORDS
-----------
Bellur & Elhilali give each sub-network a small memory of what the attended sound "looks like" from
that sub-network's point of view (Sec. IV-A). Here, every one of the 15 sub-networks gets:

- a SPEECH memory: the typical pattern of its units when it hears clean speech, and
- a TRANSIENT memory: the typical pattern when it hears a crinkle/clatter/knock on its own.
- a few "anything else" patterns learned from general sound, to soak up whatever doesn't fit.

These are found with non-negative matrix factorisation (NMF), the same tool Bellur used: it finds a
few non-negative "prototype" patterns that best explain a large pile of activity recordings.

The memories are built once, from training data only, after the belief network is pretrained.
Later, the attention gate compares live activity against them.

Design choice to watch: Bellur used a single prototype per class. Speech is very varied (vowels,
fricatives, silence), so we allow K speech prototypes (default 4) and check whether 1 is enough.

IMPLEMENTATION NOTES (status: IMPLEMENTED)
------------------------------------------
Sources (training data only): speech = VCTK TRAIN talkers (stage-1 pack), transients = TRAIN clips of the
seen transient classes, "anything else" = pretraining sounds + music.
- Speech activity is taken only from non-silent frames (within 40 dB of the crop's loudest frame).
- Transient activity is taken from the first 300 ms of each isolated clip (peak at 50 ms).
- NMF: Lee & Seung multiplicative updates on the Frobenius loss (as Bellur), small L1 on activations,
  bases normalised to unit length. K_speech = 4, K_transient = 2, K_free = CFG.attn.n_free_bases.
- Go/no-go report (stage 3): on HELD-OUT speech (val talkers) and held-out transients (val clips), the
  online projection (attention_feedback.beliefs) should give a higher speech share on speech than on
  transients. `separability_report` prints that gap per LSN.

USAGE:  python local_memory.py      # after dbn_pretrain.py; ~5–10 min on CPU -> runs/dbn/memories.pkl
"""

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dbn_distributed as dd
import frontend_gammatone as fe
from audio_store import load_pack
from config import CFG

K_SPEECH, K_TRANS = 4, 2
FS = CFG.audio.sample_rate


def _level(x):
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-8) * CFG.data.speech_rms


def speech_crops(pack, n, L=2 * FS, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, L), np.float32)
    for r, i in enumerate(rng.choice(len(pack), n)):
        c = pack.clip(i); s = rng.integers(0, max(1, len(c) - L + 1)); seg = c[s: s + L]
        out[r, : len(seg)] = _level(seg)
    return out


def transient_clips(pack, L=int(0.3 * FS)):
    out = np.zeros((len(pack), L), np.float32)
    for i in range(len(pack)):
        c = pack.clip(i)[:L]; out[i, : len(c)] = _level(c)
    return out


def active_base_frames(x: jnp.ndarray, below_db: float = 40.0) -> jnp.ndarray:
    """(B, N) -> (B, T) bool: base frames within `below_db` of the loudest frame in each row."""
    e = 10 * jnp.log10(fe.band_energies(fe.stft(x)).sum(-1) + 1e-12)
    return e > (e.max(axis=1, keepdims=True) - below_db)


def collect(dbn_params, stats, x: np.ndarray, frame_mask_fn, batch: int = 32) -> dict:
    """Run the frozen DBN (no attention) on waveforms x; return {lsn: (U, N_frames) activity on kept frames}."""
    from dbn_pretrain import dbn_features
    fwd = jax.jit(lambda xb: {l: o["F_native"] for l, o in dd.forward(dbn_params, dbn_features(xb, stats)).items()})
    acc = {lsn: [] for lsn in dbn_params}
    for i in range(0, len(x), batch):
        xb = jnp.asarray(x[i: i + batch])
        out = fwd(xb)
        keep_base = frame_mask_fn(xb)                                   # (B, T) bool at base rate
        for lsn, F in out.items():
            R = dd.rate_of(*lsn); Tn = F.shape[-1]
            keep = keep_base[:, : Tn * R].reshape(keep_base.shape[0], Tn, R).any(-1)
            acc[lsn].append(np.asarray(F).transpose(1, 0, 2)[:, np.asarray(keep)])
    return {lsn: np.concatenate(v, axis=1) for lsn, v in acc.items()}


def nmf(A: np.ndarray, k: int, n_iter: int = 200, l1: float = 0.01, max_frames: int = 100_000, seed: int = 0) -> np.ndarray:
    """Return W (U, k), columns unit-norm, with A ~= W H, W, H >= 0."""
    rng = np.random.default_rng(seed)
    if A.shape[1] > max_frames:
        A = A[:, rng.choice(A.shape[1], max_frames, replace=False)]
    A = jnp.asarray(A) + 1e-9
    W = jnp.asarray(rng.uniform(0.1, 1.0, (A.shape[0], k)).astype(np.float32))
    H = jnp.asarray(rng.uniform(0.1, 1.0, (k, A.shape[1])).astype(np.float32))

    def body(_, WH):
        W, H = WH
        H = H * (W.T @ A) / (W.T @ W @ H + l1 + 1e-9)
        W = W * (A @ H.T) / (W @ (H @ H.T) + 1e-9)
        n = jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-9
        return W / n, H * n.T
    W, H = jax.lax.fori_loop(0, n_iter, body, (W, H))
    return np.asarray(W)


def build_memories(dbn_params: dict, stats: dict, cache_dir: Path = CFG.data.cache_dir, n_speech_crops: int = 200) -> dict:
    sp = speech_crops(load_pack(Path(cache_dir) / "speech_train.npz"), n_speech_crops)
    tr = transient_clips(load_pack(Path(cache_dir) / "transient_train.npz"))
    other = []
    for g in ("sounds", "music"):
        p = Path(cache_dir) / f"pretrain_{g}.npz"
        if p.exists():
            other.append(speech_crops(load_pack(p), n_speech_crops // 2, seed=1))
    A_s = collect(dbn_params, stats, sp, active_base_frames)
    A_t = collect(dbn_params, stats, tr, lambda xb: jnp.ones(xb.shape[:1] + (fe.n_frames(xb.shape[1]),), bool))
    A_o = collect(dbn_params, stats, np.concatenate(other), active_base_frames) if other else None
    mem = {}
    for lsn in dbn_params:
        mem[lsn] = {"W_speech": nmf(A_s[lsn], K_SPEECH), "W_trans": nmf(A_t[lsn], K_TRANS),
                    "W_free": nmf(A_o[lsn], CFG.attn.n_free_bases) if A_o is not None else None}
        print(f"  LSN{lsn[0]}{lsn[1]}: memories from {A_s[lsn].shape[1]} speech / {A_t[lsn].shape[1]} transient frames")
    return mem


def separability_report(dbn_params: dict, stats: dict, memories: dict, cache_dir: Path = CFG.data.cache_dir) -> dict:
    """Held-out check: does the online projection give speech a bigger share on speech than on transients?"""
    import attention_feedback as af
    sp = speech_crops(load_pack(Path(cache_dir) / "speech_val.npz"), 64, seed=5)
    tr = transient_clips(load_pack(Path(cache_dir) / "transient_val.npz"))
    rep = {}
    for name, x, mask_fn in (("speech", sp, active_base_frames),
                             ("transient", tr, lambda xb: jnp.ones(xb.shape[:1] + (fe.n_frames(xb.shape[1]),), bool))):
        A = collect(dbn_params, stats, x, mask_fn)
        for lsn, F in A.items():
            bel = np.asarray(af.beliefs(jnp.asarray(F)[None], af.memory_matrix(memories[lsn])))[0]   # (3, N)
            share = bel[0] / (bel[0] + bel[1] + 1e-9)
            rep.setdefault(f"LSN{lsn[0]}{lsn[1]}", {})[f"speech_share_on_{name}"] = float(share.mean())
    for k, v in rep.items():
        v["gap"] = v["speech_share_on_speech"] - v["speech_share_on_transient"]
    return rep


if __name__ == "__main__":
    from dbn_pretrain import DBN_DIR, load_dbn
    params, stats = load_dbn()
    memories = build_memories(params, stats)
    pickle.dump(jax.tree_util.tree_map(lambda a: a, memories), open(DBN_DIR / "memories.pkl", "wb"))
    rep = separability_report(params, stats, memories)
    (DBN_DIR / "memory_separability.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(f"  {k}: speech share on speech {v['speech_share_on_speech']:.2f}  on transients "
              f"{v['speech_share_on_transient']:.2f}  gap {v['gap']:+.2f}")
    good = sum(v["gap"] > 0.2 for v in rep.values())
    print(f"GO/NO-GO: {good}/{len(rep)} sub-networks separate speech from transients (gap > 0.2) -> "
          f"{'GO' if good >= 3 else 'NO-GO: attention has little to work with'}")
