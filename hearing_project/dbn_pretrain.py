# Hearing project: unsupervised pretraining of the belief network - Jake Abraham
"""
PLAIN WORDS
-----------
Trains the 15 sub-networks *without labels*, one at a time from the bottom of the tree up, on a broad
mix of everyday audio (~3 h speech, ~4 h household/environmental sounds, ~2 h music) — never on the
test talkers or on any transient category. This mirrors Bellur & Elhilali: the belief network is
meant to be a general-purpose description of natural sound, not something tuned to the transient
task. That generality is the reason it is expected to cope better with crinkles it has never heard.

Each sub-network is trained for up to 10 passes over the data, and stops early once its ability to
re-create held-out audio stops improving. Every finished sub-network is saved immediately, so if the
run is interrupted, re-running the script picks up where it left off.

When finished, it saves diagnostics: the training curves, example templates from the first
sub-network (they should look like little spectro-temporal patches), and how strongly each
sub-network fires for clean speech versus isolated transients — a first hint of whether the network
separates the two, which is what the attention stage will rely on.

USAGE
-----
    python data_download.py --only pretrain     # once: builds the pretraining audio packs
    python dbn_pretrain.py                      # ~2–3 h on the sandbox CPU; resumable
    python dbn_pretrain.py --smoke              # 1-minute check on synthetic audio

IMPLEMENTATION NOTES (status: IMPLEMENTED)
------------------------------------------
- Batches: 32 random 2-s crops, sampled in proportion to duration across speech / sounds / music.
  3% of clips in each group are held out for early stopping.
- Features: z-scored 64-band log-gammatone energies (same front end as everything else), with
  normalisation statistics computed on the pretraining audio and saved with the network.
- Each LSN's input = frozen upward pass through its already-trained ancestors (no attention).
- Early stopping: held-out mean-field reconstruction error, checked every 250 steps; stop after
  4 checks without a >0.5% improvement (min 1,000 steps, max epochs_per_layer passes).
- STILL TODO (optional diagnostics): projecting upper-layer bases to the filterbank domain and the
  rate–scale modulation map (Bellur Fig. 4).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dbn_crbm as cr
import dbn_distributed as dd
import frontend_gammatone as fe
from audio_store import load_pack
from config import CFG, DATA
from optim import adam_init, adam_update, make_schedule

DBN_DIR = CFG.out_dir / "dbn"
GROUPS = ("speech", "sounds", "music")
CROP = CFG.mask.seq_len_frames * CFG.audio.hop_len          # 2-s crops


###########################################################
#                          DATA                           #
###########################################################

class CropSampler:
    """Random fixed-length crops from the pretraining packs, sampled in proportion to duration."""

    def __init__(self, cache_dir: Path = CFG.data.cache_dir, held_out: float = 0.03, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.clips = {"train": [], "heldout": []}
        for g in GROUPS:
            path = Path(cache_dir) / f"pretrain_{g}.npz"
            if not path.exists():
                continue
            pack = load_pack(path)
            idx = rng.permutation(len(pack))
            n_ho = max(1, int(held_out * len(pack)))
            for split, ids in (("heldout", idx[:n_ho]), ("train", idx[n_ho:])):
                self.clips[split] += [pack.clip(i) for i in ids]
        if not self.clips["train"]:
            raise FileNotFoundError("no pretrain_*.npz packs: run `python data_download.py --only pretrain`")
        self.weights = {k: np.array([len(c) for c in v], float) / sum(len(c) for c in v) for k, v in self.clips.items()}
        self.hours = sum(len(c) for c in self.clips["train"]) / 16_000 / 3600

    def batch(self, rng: np.random.Generator, split: str = "train", B: int = 32, L: int = CROP) -> np.ndarray:
        out = np.zeros((B, L), np.float32)
        for b, i in enumerate(rng.choice(len(self.clips[split]), size=B, p=self.weights[split])):
            c = self.clips[split][i]
            s = rng.integers(0, max(1, len(c) - L + 1))
            seg = c[s: s + L]
            out[b, : len(seg)] = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-8) * CFG.data.speech_rms   # level-normalise
        return out


def log_gammatone(x: jnp.ndarray) -> jnp.ndarray:
    """(B, N) waveform -> (B, 64, T) un-normalised log band energies."""
    return jnp.swapaxes(fe.log_features(fe.band_energies(fe.stft(x))), 1, 2)


def feature_stats(sampler: CropSampler, n: int = 32, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    L = np.concatenate([np.asarray(log_gammatone(jnp.asarray(sampler.batch(rng)))).transpose(0, 2, 1).reshape(-1, 64)
                        for _ in range(n)])
    return {"mean": L.mean(0).astype(np.float32), "std": (L.std(0) + 1e-5).astype(np.float32)}


def dbn_features(x: jnp.ndarray, stats: dict) -> jnp.ndarray:
    return (log_gammatone(x) - stats["mean"][None, :, None]) / stats["std"][None, :, None]


###########################################################
#                     LAYER-WISE TRAINING                 #
###########################################################

def ancestors(lsn: tuple) -> list:
    path, (g, z) = [], lsn
    while g > 1:
        g, z = dd.parent(g, z)
        path.append((g, z))
    return path[::-1]


def make_input_fn(lsn: tuple, stats: dict):
    path = ancestors(lsn)

    @jax.jit
    def get_input(frozen: dict, x: jnp.ndarray) -> jnp.ndarray:
        v = dbn_features(x, stats)
        for a in path:
            v = cr.encode(frozen[a], v, dd.pool_of(*a))
        return v
    return get_input


def meanfield_recon(p: dict, v: jnp.ndarray, vtype: str, pool: int) -> jnp.ndarray:
    T = (v.shape[-1] // pool) * pool
    v = v[..., :T]
    det, _ = cr.prob_max_pool(cr.hidden_preact(p, v), pool)
    return jnp.mean((v - cr.visible_mean(p, det, vtype)) ** 2)


def train_lsn(lsn, frozen, sampler, stats, max_steps, batch=32, eval_every=250, patience=4, min_steps=1000,
              seed=0, log=print):
    vtype, pool = dd.visible_type(*lsn), dd.pool_of(*lsn)
    get_input = make_input_fn(lsn, stats)
    p = cr.init_crbm(jax.random.PRNGKey(seed + 17 * lsn[0] + lsn[1]), dd.input_channels(*lsn))
    lr_fn = make_schedule(CFG.dbn.lr, max_steps)
    opt = adam_init(p)

    @jax.jit
    def step(p, opt, v, key):
        (loss, m), g = jax.value_and_grad(lambda q: cr.cd_loss(q, v, key, vtype, pool), has_aux=True)(p)
        p, opt = adam_update(p, g, opt, lr_fn)
        return p, opt, loss, m

    eval_fn = jax.jit(lambda p, v: meanfield_recon(p, v, vtype, pool))
    rng_ho = np.random.default_rng(999)
    held = [get_input(frozen, jnp.asarray(sampler.batch(rng_ho, "heldout", batch))) for _ in range(6)]
    rng = np.random.default_rng(seed + 1000 * lsn[0] + lsn[1])
    best, best_p, since, hist, t0 = np.inf, p, 0, [], time.time()
    for it in range(1, max_steps + 1):
        v = get_input(frozen, jnp.asarray(sampler.batch(rng, "train", batch)))
        p, opt, loss, m = step(p, opt, v, jax.random.PRNGKey(it))
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f"LSN{lsn} diverged at step {it}")
        if it % eval_every == 0 or it == max_steps:
            ho = float(np.mean([eval_fn(p, h) for h in held]))
            row = dict(step=it, held_out_recon=ho, train_recon=float(m["recon_mse"]), mean_act=float(m["mean_act"]),
                       pooled_act=float(m["pooled_act"]), dead_frac=float(m["dead_frac"]), minutes=(time.time() - t0) / 60)
            hist.append(row)
            if ho < best * 0.995:
                best, best_p, since = ho, p, 0
            else:
                since += 1
            log(f"  LSN{lsn[0]}{lsn[1]} step {it:5d}  held-out recon {ho:.4f}  act {row['mean_act']:.3f}  "
                f"dead {row['dead_frac']:.2f}  ({row['minutes']:.1f} min)")
            if it >= min_steps and since >= patience:
                break
    return best_p, hist


def pretrain_dbn(sampler: CropSampler, out_dir: Path = DBN_DIR, batch: int = 32, max_epochs: float = CFG.dbn.epochs_per_layer,
                 seed: int = 0, only: tuple = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "feature_stats.npz"
    if stats_path.exists():
        z = np.load(stats_path); stats = {"mean": z["mean"], "std": z["std"]}
    else:
        stats = feature_stats(sampler); np.savez(stats_path, **stats)
    stats = {k: jnp.asarray(v) for k, v in stats.items()}
    steps_per_epoch = int(sampler.hours * 3600 * 16_000 / (batch * CROP))
    max_steps = max(1, int(max_epochs * steps_per_epoch))
    logf = open(out_dir / "pretrain_log.txt", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    log(f"pretraining on {sampler.hours:.2f} h; {steps_per_epoch} steps/epoch; max {max_steps} steps per LSN")
    params, history = {}, {}
    for lsn in dd.lsn_ids():
        ckpt = out_dir / f"lsn{lsn[0]}{lsn[1]}.pkl"
        if ckpt.exists():
            d = pickle.load(open(ckpt, "rb"))
            params[lsn] = jax.tree_util.tree_map(jnp.asarray, d["params"]); history[lsn] = d["history"]
            log(f"LSN{lsn[0]}{lsn[1]}: loaded checkpoint"); continue
        if only and lsn not in only:
            break
        log(f"LSN{lsn[0]}{lsn[1]}: training (input {dd.input_channels(*lsn)} ch, pool {dd.pool_of(*lsn)}, "
            f"{dd.visible_type(*lsn)} visible, rate {dd.rate_of(*lsn)})")
        p, hist = train_lsn(lsn, params, sampler, stats, max_steps, batch, seed=seed, log=log)
        params[lsn], history[lsn] = p, hist
        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, p), "history": hist}, open(ckpt, "wb"))
    return params, stats, history


def load_dbn(out_dir: Path = DBN_DIR) -> tuple:
    """Load every trained LSN + the feature stats. Returns (params, stats)."""
    params = {}
    for lsn in dd.lsn_ids():
        ckpt = Path(out_dir) / f"lsn{lsn[0]}{lsn[1]}.pkl"
        if ckpt.exists():
            params[lsn] = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(ckpt, "rb"))["params"])
    z = np.load(Path(out_dir) / "feature_stats.npz")
    return params, {"mean": jnp.asarray(z["mean"]), "std": jnp.asarray(z["std"])}


###########################################################
#                       DIAGNOSTICS                       #
###########################################################

def diagnostics(params: dict, stats: dict, history: dict, out_dir: Path = DBN_DIR, synthetic: bool = False):
    """Training curves, LSN11 templates, and speech-vs-transient firing per LSN (from stage-1 packs)."""
    import data_mixing as dm

    report = {}
    try:
        corpus = dm.load_corpus(DATA / "synthetic_cache" if synthetic else CFG.data.cache_dir)
        rng = np.random.default_rng(0)
        def stack(pack, n):
            L = 16_000
            out = np.zeros((n, L), np.float32)
            for r, i in enumerate(rng.choice(len(pack), n, replace=len(pack) < n)):
                c = pack.clip(i)[:L]; out[r, : len(c)] = c / (np.sqrt(np.mean(c ** 2)) + 1e-8) * CFG.data.speech_rms
            return jnp.asarray(out)
        sp, tr = stack(corpus.speech["val"], 32), stack(corpus.transients["train"], 32)
        f_sp = dd.forward(params, dbn_features(sp, stats)); f_tr = dd.forward(params, dbn_features(tr, stats))
        for lsn in params:
            a, b = f_sp[lsn]["F_native"].mean(axis=(0, 2)), f_tr[lsn]["F_native"].mean(axis=(0, 2))
            cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-9))
            report[f"LSN{lsn[0]}{lsn[1]}"] = dict(speech_mean_firing=float(a.mean()), transient_mean_firing=float(b.mean()),
                                                  cosine_speech_vs_transient=cos)
    except FileNotFoundError:
        pass
    (out_dir / "diagnostics.json").write_text(json.dumps(report, indent=1))
    plot_diagnostics(params, history, report, out_dir, synthetic)
    return report


def plot_diagnostics(params: dict, history: dict, report: dict, out_dir: Path = DBN_DIR, synthetic: bool = False):
    """Draw diagnostics.png: training curves, speech-vs-transient similarity per LSN, LSN11 templates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from money_plot import INK, SURFACE, _style_axes

    fig = plt.figure(figsize=(15, 10), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.42)
    top = outer[0].subgridspec(1, 2, wspace=0.22)
    ax = fig.add_subplot(top[0]); _style_axes(ax)
    layer_colour = {1: "#2a78d6", 2: "#1baf7a", 3: "#eb6834", 4: "#7a5bd0"}
    seen = set()
    for lsn, h in history.items():
        if h:
            r0 = h[0]["held_out_recon"]
            lab = None if lsn[0] in seen else f"layer {lsn[0]} ({sum(1 for l in history if l[0] == lsn[0])} LSNs)"
            seen.add(lsn[0])
            ax.plot([x["step"] for x in h], [x["held_out_recon"] / r0 for x in h], lw=1.4, alpha=0.85,
                    color=layer_colour.get(lsn[0], INK), label=lab)
    ax.set_title("Held-out reconstruction error (relative to first check)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("training step"); ax.legend(fontsize=9, frameon=False)
    ax = fig.add_subplot(top[1]); _style_axes(ax)
    if report:
        names = list(report)
        ax.bar(np.arange(len(names)), [report[n]["cosine_speech_vs_transient"] for n in names], color="#2a78d6")
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=8)
        ax.set_ylim(0, 1)
    ax.set_title("Similarity of firing pattern: speech vs transients\n(lower = the sub-network tells them apart)",
                 loc="left", fontsize=11, color=INK)
    if (1, 1) in params:
        K = np.asarray(params[(1, 1)]["K"])                     # (U, 64, W)
        order = np.argsort(-np.abs(K).reshape(K.shape[0], -1).max(1))[:16]
        grid = outer[1].subgridspec(2, 8, wspace=0.12, hspace=0.12)
        for j, u in enumerate(order):
            a = fig.add_subplot(grid[j // 8, j % 8])
            lim = np.abs(K[u]).max()
            a.imshow(K[u], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_title("LSN11 templates (rows: 64 gammatone bands, low→high; columns: 12 frames = 30 ms; "
                            "red = excitatory)", loc="left", fontsize=11, color=INK, pad=8)
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC SMOKE TEST", ha="center", fontsize=40, color="#b8322f", alpha=0.25, rotation=20)
    fig.savefig(out_dir / "diagnostics.png", dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def replot_diagnostics(out_dir: Path = DBN_DIR):
    """Re-draw diagnostics.png from saved checkpoints + diagnostics.json (no network evaluation)."""
    params, history = {}, {}
    for lsn in dd.lsn_ids():
        ckpt = Path(out_dir) / f"lsn{lsn[0]}{lsn[1]}.pkl"
        if ckpt.exists():
            d = pickle.load(open(ckpt, "rb"))
            params[lsn], history[lsn] = d["params"], d["history"]
    report = json.loads((Path(out_dir) / "diagnostics.json").read_text())
    plot_diagnostics(params, history, report, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic audio, 2 LSNs, a few steps")
    ap.add_argument("--epochs", type=float, default=CFG.dbn.epochs_per_layer)
    ap.add_argument("--replot", action="store_true", help="re-draw diagnostics.png from saved checkpoints")
    a = ap.parse_args()
    if a.replot:
        replot_diagnostics()
    elif a.smoke:
        import synthetic_audio
        from audio_store import save_pack
        syn = synthetic_audio.build()
        sp = load_pack(syn / "speech_train.npz"); tr = load_pack(syn / "transient_train.npz")
        save_pack(syn / "pretrain_speech.npz", [sp.clip(i) for i in range(len(sp))], sp.meta, synthetic=True)
        save_pack(syn / "pretrain_sounds.npz", [tr.clip(i) for i in range(len(tr))], tr.meta, synthetic=True)
        sampler = CropSampler(syn)
        out = CFG.out_dir / "smoke" / "dbn"
        import shutil; shutil.rmtree(out, ignore_errors=True)
        params, stats, hist = pretrain_dbn(sampler, out, batch=8, max_epochs=0.0, only=((1, 1), (2, 1), (2, 2)))
        # max_epochs=0 -> 1 step; run a few real steps for the first LSNs instead:
        for lsn in ((1, 1), (2, 1), (2, 2)):
            p, h = train_lsn(lsn, params, sampler, stats, max_steps=40, batch=8, eval_every=20, min_steps=40)
            params[lsn], hist[lsn] = p, h
        print(json.dumps(diagnostics(params, stats, hist, out, synthetic=True), indent=1))
    else:
        sampler = CropSampler()
        params, stats, hist = pretrain_dbn(sampler, max_epochs=a.epochs)
        print(json.dumps(diagnostics(params, stats, hist), indent=1))
