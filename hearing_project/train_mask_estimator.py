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

IMPLEMENTATION STATUS: IMPLEMENTED for every learned variant (C, D, E, F, G, H). For F/G/H the belief
network and memories are frozen and passed in as `aux`; only readouts, selector, LayerNorm and LSTMs learn.
Stage-5 fine-tuning (`finetune`) is IMPLEMENTED: it unfreezes the belief network (lr 1e-5) and the attention gate's
alpha / beta / temperature (lr 1e-4) and trains them end to end with the mask network (lr 1e-4), with a frozen
control run of the same length. Choices made: early stopping on validation mask error (as in stages 1-4, cheaper
than ESTOI); no sparsity regulariser (the small learning rate bounds weight drift; see `finetune`).
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
          val_every: int = 100, tag: str = None, dbn_dir: Path = None) -> Path:
    assert variant in mv.NEURAL, f"variant {variant} is not a learned model"
    mode = mv.MODES[variant]
    run_dir = run_dir_for(corpus, tag)
    out_dir = run_dir / variant / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = get_stats(corpus, run_dir)
    recipes = get_recipes(corpus, run_dir)

    val = [crop_around_peak(dm.realise(corpus, r)) for r in recipes["rows"] if r["set"] == "val"]
    vx, vs, vv = (jnp.asarray(np.stack([m[k] for m in val])) for k in ("x", "s", "v"))
    val_irm = fe.ideal_ratio_mask(vs, vv)

    params = mv.init_params(variant, jax.random.PRNGKey(seed), dbn_dir)
    aux = mv.build_aux(variant, stats, dbn_dir)
    lr_fn = make_schedule(CFG.mask.lr, n_steps)
    opt = adam_init(params)
    mse = lambda params, aux, x, irm: jnp.mean((mv.predict(variant, params, aux, x) - irm) ** 2)

    @jax.jit
    def train_step(params, opt, aux, x, s, v):
        loss, grads = jax.value_and_grad(mse)(params, aux, x, fe.ideal_ratio_mask(s, v))
        grads, gnorm = clip_by_global_norm(grads, CFG.mask.grad_clip)
        params, opt = adam_update(params, grads, opt, lr_fn)
        return params, opt, loss, gnorm

    val_chunk = jax.jit(mse)

    def val_loss(params):
        return np.mean([float(val_chunk(params, aux, vx[i: i + 25], val_irm[i: i + 25])) for i in range(0, vx.shape[0], 25)])

    batches = dm.batch_iterator(corpus, "train", batch_size=batch_size, seed=seed)
    print(f"[{variant}] mode={mode} params={n_params(params):,} steps={n_steps} batch={batch_size} "
          f"-> {out_dir}{'  (SYNTHETIC SMOKE TEST)' if corpus.synthetic else ''}")
    best, log, t0, run_loss = np.inf, [], time.time(), []
    for step in range(1, n_steps + 1):
        b = next(batches)
        params, opt, loss, gnorm = train_step(params, opt, aux, jnp.asarray(b["x"]), jnp.asarray(b["s"]), jnp.asarray(b["v"]))
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


def finetune(variant: str, corpus: dm.Corpus, seed: int = 0, n_steps: int = 2000, batch_size: int = 16,
             unfreeze: bool = True, lr_net: float = 1e-4, lr_dbn: float = 1e-5, val_every: int = 100,
             tag: str = None, data_seed_offset: int = 100, start_label: str = None, label: str = None) -> Path:
    """Stage 5. Start from a trained checkpoint and train for `n_steps` more on NEW data (data seed + 100).

    unfreeze=True  -> "<V>_ft": the belief network (lr 1e-5) and the attention gate's alpha / beta / temperature
                      (lr 1e-4) learn together with the mask network (lr 1e-4), end to end from the mask loss.
    unfreeze=False -> "<V>_cont": the control. Identical extra steps, data and learning rate, but the belief network
                      stays frozen. Comparing the two isolates "unfreezing" from "just more training".
    The step-0 model is evaluated first and kept unless validation improves, so fine-tuning can never make the saved
    checkpoint worse on validation. No extra regulariser: at lr 1e-5 a weight can move at most ~0.02 in 2,000 steps
    (DBN weights average |0.02|), which keeps the pretrained features in place.

    Stage 6 (longer training) reuses this to continue a frozen run: start_label="H_cont" loads runs/.../H_cont/seed{k}
    instead of H/seed{k}, and label="H_long" names the output. `variant` still says which model it is."""
    run_dir = run_dir_for(corpus, tag)
    start = mv.Trained.load(run_dir / (start_label or variant) / f"seed{seed}" / "best.pkl")
    unfreeze = unfreeze and variant in mv.DBN_VARIANTS
    label = label or (f"{variant}_ft" if unfreeze else f"{variant}_cont")
    out_dir = run_dir / label / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    recipes = get_recipes(corpus, run_dir)
    val = [crop_around_peak(dm.realise(corpus, r)) for r in recipes["rows"] if r["set"] == "val"]
    vx, vs, vv = (jnp.asarray(np.stack([m[k] for m in val])) for k in ("x", "s", "v"))
    val_irm = fe.ideal_ratio_mask(vs, vv)

    aux0 = start.aux
    fast = {"net": start.params}                                      # lr_net
    slow = {}                                                         # lr_dbn
    if unfreeze:
        slow["dbn"] = aux0["dbn"]
        if mv.DBN_VARIANTS[variant]["attn"]:
            fast["attn"] = mv.attn_params_init()

    def assemble(fast, slow):
        aux = {**aux0, **slow}
        if "attn" in fast:
            aux["attn"] = fast["attn"]
        return fast["net"], aux

    def mse(fast, slow, x, irm):
        p, aux = assemble(fast, slow)
        return jnp.mean((mv.predict(variant, p, aux, x) - irm) ** 2)

    lr_f, lr_s = make_schedule(lr_net, n_steps), make_schedule(lr_dbn, n_steps)
    opt_f, opt_s = adam_init(fast), adam_init(slow)

    @jax.jit
    def train_step(fast, slow, opt_f, opt_s, x, s, v):
        loss, (gf, gs) = jax.value_and_grad(mse, argnums=(0, 1))(fast, slow, x, fe.ideal_ratio_mask(s, v))
        (gf, gs), gnorm = clip_by_global_norm((gf, gs), CFG.mask.grad_clip)
        fast, opt_f = adam_update(fast, gf, opt_f, lr_f)
        if unfreeze:
            slow, opt_s = adam_update(slow, gs, opt_s, lr_s)
        return fast, slow, opt_f, opt_s, loss, gnorm

    val_chunk = jax.jit(mse)

    def val_loss(fast, slow):
        return np.mean([float(val_chunk(fast, slow, vx[i: i + 25], val_irm[i: i + 25])) for i in range(0, vx.shape[0], 25)])

    def save(fast, slow):
        extra = start.extra                                           # keep a fine-tuned start's network when frozen
        if unfreeze:
            extra = {"dbn": slow["dbn"], **({"attn": fast["attn"]} if "attn" in fast else {})}
        mv.Trained(variant, fast["net"], start.stats, extra).save(out_dir / "best.pkl")

    n_slow = n_params(slow)
    print(f"[{label}] from {start_label or variant}/seed{seed}  trainable={n_params(fast) + n_slow:,} (belief network {n_slow:,})  "
          f"steps={n_steps} batch={batch_size} lr_net={lr_net} lr_dbn={lr_dbn if unfreeze else '-'} -> {out_dir}", flush=True)
    best = float(val_loss(fast, slow))
    save(fast, slow)
    log, t0, run_loss = [dict(step=0, train_loss=np.nan, val_loss=best, grad_norm=np.nan, minutes=0.0)], time.time(), []
    print(f"  step     0  val {best:.4f}  (starting checkpoint)", flush=True)
    batches = dm.batch_iterator(corpus, "train", batch_size=batch_size, seed=seed + data_seed_offset)
    for step in range(1, n_steps + 1):
        b = next(batches)
        fast, slow, opt_f, opt_s, loss, gnorm = train_step(fast, slow, opt_f, opt_s, jnp.asarray(b["x"]),
                                                           jnp.asarray(b["s"]), jnp.asarray(b["v"]))
        run_loss.append(float(loss))
        if step % val_every == 0 or step == n_steps:
            vl = float(val_loss(fast, slow))
            row = dict(step=step, train_loss=np.mean(run_loss), val_loss=vl, grad_norm=float(gnorm),
                       minutes=(time.time() - t0) / 60)
            log.append(row); run_loss = []
            improved = vl < best
            if improved:
                best = vl
                save(fast, slow)
            extra = ""
            if "attn" in fast:
                a = mv.attn_settings(fast["attn"])
                extra = f"  alpha {float(a['alpha']):.3f} beta {float(a['beta']):+.3f} T {float(a['temperature']):.3f}"
            print(f"  step {step:5d}  train {row['train_loss']:.4f}  val {vl:.4f}{'  *' if improved else ''}"
                  f"  ({row['minutes']:.1f} min){extra}", flush=True)
    with open(out_dir / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    (out_dir / "done").write_text("ok\n")
    return out_dir / "best.pkl"


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
