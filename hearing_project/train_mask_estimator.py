# Hearing project: supervised training of the mask estimator - Jake Abraham
"""
PLAIN WORDS
-----------
Teaches each learned variant to predict the ideal mask, using freshly mixed speech + transient
examples every step. Progress is checked on a fixed validation set (unseen talkers, validation
transients), and the checkpoint with the lowest validation error is kept.

How much training? Keshavarzi et al. trained on 1,600 sentences for 5 passes, i.e. roughly 6 million
2.5-ms frames. One step here is 16 two-second crops = 12,800 frames, so ~500 steps equals their budget.
The default of 2,000 steps gives 4x their budget (dynamic mixing means it never repeats an example).
On the sandbox CPU that is roughly 10–20 minutes per variant.

Every variant gets the SAME data stream (same seed) and the same number of steps, so differences
come from the model, not from luck in the data.

USAGE
-----
    python train_mask_estimator.py C            # Keshavarzi baseline (3-frame memory)
    python train_mask_estimator.py D            # same network, continuous memory
    python train_mask_estimator.py C --smoke    # 60-step run on synthetic data, just to test the code

IMPLEMENTATION STATUS: IMPLEMENTED for variants C and D. Stage-5 fine-tuning (`finetune`) is still
pseudocode: unfreeze DBN + attention alpha/beta/temperature (lr 1e-5 DBN, 1e-4 rest), keep the CRBM
sparsity penalty as a regulariser, early-stop on validation ESTOI.
"""

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import data_mixing as dm
import frontend_gammatone as fe
import mask_estimator_lstm as ml
import model_variants as mv
from config import CFG, DATA
from optim import adam_init, adam_update, clip_by_global_norm, make_schedule, n_params

L_CROP = CFG.mask.seq_len_frames * CFG.audio.hop_len


def crop_around_peak(m: dict, L: int = L_CROP) -> dict:
    """Deterministic 2-s crop of a validation mixture that contains the transient's peak."""
    peak = m["onset"] + dm.PRE_PEAK
    start = int(np.clip(peak - L // 2, 0, max(0, len(m["x"]) - L)))
    out = {}
    for k in ("x", "s", "v"):
        a = np.zeros(L, np.float32); seg = m[k][start: start + L]; a[: len(seg)] = seg; out[k] = a
    return out


def run_dir_for(corpus: dm.Corpus, tag: str = None) -> Path:
    return CFG.out_dir / (tag or ("smoke" if corpus.synthetic else "stage1"))


def get_recipes(corpus: dm.Corpus, run_dir: Path) -> dict:
    path = run_dir / "recipes.json"
    if not path.exists():
        dm.save_recipes(dm.build_recipes(corpus), path)
    return dm.load_recipes(path)


def get_stats(corpus: dm.Corpus, run_dir: Path) -> dict:
    path = run_dir / "feature_stats.npz"
    if path.exists():
        z = np.load(path)
        return {"mean": jnp.asarray(z["mean"]), "std": jnp.asarray(z["std"])}
    stats = mv.compute_feature_stats(dm.batch_iterator(corpus, "train", batch_size=16, seed=999))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.asarray(stats["mean"]), std=np.asarray(stats["std"]))
    return stats


def train(variant: str, corpus: dm.Corpus, n_steps: int = 2000, batch_size: int = 16, seed: int = 0,
          val_every: int = 100, tag: str = None) -> Path:
    assert variant in mv.NEURAL, f"variant {variant} is not trainable in stage 1"
    mode = mv.MODES[variant]
    run_dir = run_dir_for(corpus, tag)
    out_dir = run_dir / variant / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = get_stats(corpus, run_dir)
    recipes = get_recipes(corpus, run_dir)

    val = [crop_around_peak(dm.realise(corpus, r)) for r in recipes["rows"] if r["set"] == "val"]
    vx, vs, vv = (jnp.asarray(np.stack([m[k] for m in val])) for k in ("x", "s", "v"))
    val_irm = fe.ideal_ratio_mask(vs, vv)

    params = ml.init_mask_net(jax.random.PRNGKey(seed), CFG.audio.n_bands)
    lr_fn = make_schedule(CFG.mask.lr, n_steps)
    opt = adam_init(params)

    @jax.jit
    def train_step(params, opt, x, s, v):
        u, irm = mv.features(x, stats), fe.ideal_ratio_mask(s, v)
        loss, grads = jax.value_and_grad(ml.loss_fn)(params, u, irm, mode)
        grads, gnorm = clip_by_global_norm(grads, CFG.mask.grad_clip)
        params, opt = adam_update(params, grads, opt, lr_fn)
        return params, opt, loss, gnorm

    @jax.jit
    def val_loss(params):
        losses = [ml.loss_fn(params, mv.features(vx[i: i + 50], stats), val_irm[i: i + 50], mode)
                  for i in range(0, vx.shape[0], 50)]
        return jnp.mean(jnp.stack(losses))

    batches = dm.batch_iterator(corpus, "train", batch_size=batch_size, seed=seed)
    print(f"[{variant}] mode={mode} params={n_params(params):,} steps={n_steps} batch={batch_size} "
          f"-> {out_dir}{'  (SYNTHETIC SMOKE TEST)' if corpus.synthetic else ''}")
    best, log, t0, run_loss = np.inf, [], time.time(), []
    for step in range(1, n_steps + 1):
        b = next(batches)
        params, opt, loss, gnorm = train_step(params, opt, jnp.asarray(b["x"]), jnp.asarray(b["s"]), jnp.asarray(b["v"]))
        run_loss.append(float(loss))
        if step % val_every == 0 or step == n_steps:
            vl = float(val_loss(params))
            row = dict(step=step, train_loss=np.mean(run_loss), val_loss=vl, grad_norm=float(gnorm),
                       minutes=(time.time() - t0) / 60)
            log.append(row); run_loss = []
            improved = vl < best
            if improved:
                best = vl
                mv.Trained(variant, params, stats).save(out_dir / "best.pkl")
            print(f"  step {step:5d}  train {row['train_loss']:.4f}  val {vl:.4f}{'  *' if improved else ''}"
                  f"  ({row['minutes']:.1f} min)", flush=True)
    with open(out_dir / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    return out_dir / "best.pkl"


def finetune(variant: str = "H", seed: int = 0, n_steps: int = 10_000):
    raise NotImplementedError("stage 5: needs the belief network")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(mv.NEURAL))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 60 steps")
    a = ap.parse_args()
    if a.smoke:
        import synthetic_audio
        corpus = dm.load_corpus(synthetic_audio.build())
        train(a.variant, corpus, n_steps=60, batch_size=8, seed=a.seed, val_every=20)
    else:
        train(a.variant, dm.load_corpus(), n_steps=a.steps, batch_size=a.batch, seed=a.seed)
