# Hearing project: the experiment grid and ablations - Jake Abraham
"""
PLAIN WORDS
-----------
The "do everything in order" script. It trains all variants, scores them on every test condition,
runs the ablations that explain *why* the proposed model works (or doesn't), and writes the tables
that money_plot.py turns into the final figure.

Test conditions (all built once with a fixed seed):
  - seen transient types / unseen transient types / grandma's crinkling (never trained on)
  - STR from −5 dB (mild) to −30 dB (very loud transient)
  - quiet background, or babble at 10 dB and 5 dB SNR (Keshavarzi's untested limitation)

Ablations (each answers one question):
  1. Remove one sub-network at a time (zero it at test)  -> which views matter for transients?
  2. Attention strength: alpha in {1.0 (off), 0.85, 0.7, 0.5}; threshold beta sweep
  3. Number of speech memories: 1 (Bellur) vs 4
  4. Keep vs drop the "rejected" stream                   -> does keeping transient evidence help?
  5. Stream selection on/off
  6. Slim (48 units) vs Bellur-size (300 units) belief network
  7. CRBM vs sparse autoencoder DBN backend
  8. Frozen vs fine-tuned belief network

USAGE (IMPLEMENTED: stage 1, stage 3, stage 4 incl. scoring, paired statistics and the money plot)
------------------------------------------------------------
    python run_experiments.py stage1 --smoke          # 2-minute end-to-end check on synthetic data
    python run_experiments.py stage1                  # real data: trains C and D, scores A B C D I
    python run_experiments.py stage1 --n 300          # all 300 test sentences per cell (slower)
    python run_experiments.py stage3                  # memories + go/no-go (after dbn_pretrain.py)
    python run_experiments.py stage4                  # trains E F G H, scores all nine, money plot
    python run_experiments.py refresh4                # add F, G to the trade-off sweep; redo verdict + money plot
Outputs in runs/stage1/ (shared experiment folder: recipes, feature stats, all trained models, tables, figures)

PSEUDOCODE FOR LATER STAGES
---------------------------
main():
    data_mixing.build_fixed_test_set(seed=1234)                  # once
    stage 1: for v in C, D: train(v, seeds 0..2); score; GO/NO-GO: C beats A and B on STOI, below I
    stage 2: dbn_pretrain.pretrain_dbn(); diagnostics
    stage 3: local_memory.build_memories(); memory_separability_report -> GO/NO-GO
    stage 4: E sized to H; for v in E, F, G, H: train(v, seeds 0..2)
    stage 5: finetune("H")
    stage 6: for every variant x condition: evaluate_metrics.score_one over test set
             tradeoff_curve for A, B, C, E, H, I at STR = −15 dB, unseen transients (money plot left)
             STOI vs STR for unseen transients (money plot right)
             ablations 1–8 on validation set only (no test-set tuning!)
             enhance_stream.check_equivalence + measure for C and H
    write runs/results/*.csv ; call money_plot.make_money_plot(from_results=True)

Rule: every hyperparameter choice (alpha, beta, memory count, exponent) is made on VALIDATION data.
      The test set is touched once, at the end.
"""


import argparse
from pathlib import Path

import numpy as np

import data_mixing as dm
import evaluate_metrics as em
import model_variants as mv
import train_mask_estimator as tm
from config import CFG, VARIANTS

STAGE1_VARIANTS = ("A", "B", "C", "D", "I")


###########################################################
#                        STAGE 1                          #
###########################################################

def select_test_rows(recipes: dict, n_per_cell: int, seed: int = 0) -> list:
    """Same subset of test sentences in every (condition, STR) cell, so cells are directly comparable."""
    rows = [r for r in recipes["rows"] if r["set"] == "test"]
    n_sent = 1 + max(r["speech_idx"] for r in rows)
    keep = set(np.random.default_rng(seed).permutation(n_sent)[:n_per_cell].tolist())
    return [r for r in rows if r["speech_idx"] in keep]


def evaluate(corpus, run_dir: Path, rows: list, models: dict, variants=STAGE1_VARIANTS, workers: int = 4) -> list:
    mixes = [dm.realise(corpus, r) for r in rows]
    for m, r in zip(mixes, rows):
        m.update(speech_idx=r["speech_idx"], trans=r["trans"], trans_idx=r["trans_idx"],
                 trans_label=corpus.transients[r["trans"]].meta[r["trans_idx"]].get("label"),
                 talker=corpus.speech["test"].meta[r["speech_idx"]].get("talker"))
    results = []
    for v in variants:
        outs = mv.process_batch(v, mixes, models.get(v))
        items = [dict(m, **o, variant=v) for m, o in zip(mixes, outs)]
        for it in items:
            it.pop("mask", None)
        print(f"  scoring {v} ({VARIANTS[v][0]}) on {len(items)} mixtures ...", flush=True)
        results += em.score_many(items, workers)
    return results


def stage1(smoke: bool = False, n_per_cell: int = 100, steps: int = 2000, batch: int = 16, seed: int = 0,
           retrain: bool = False):
    if smoke:
        import synthetic_audio
        corpus = dm.load_corpus(synthetic_audio.build())
        steps, batch, n_per_cell = 60, 8, 10
    else:
        corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    recipes = tm.get_recipes(corpus, run_dir)
    models = {}
    for v in ("C", "D"):
        ckpt = run_dir / v / f"seed{seed}" / "best.pkl"
        if retrain or not ckpt.exists():
            tm.train(v, corpus, n_steps=steps, batch_size=batch, seed=seed, val_every=20 if smoke else 100)
        models[v] = mv.Trained.load(ckpt)
    rows = select_test_rows(recipes, n_per_cell)
    results = evaluate(corpus, run_dir, rows, models)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "results_per_utterance.csv", index=False)
    summary = em.aggregate(results)
    summary.to_csv(run_dir / "summary.csv", index=False)
    verdict = go_no_go(summary)
    (run_dir / "go_no_go.txt").write_text(verdict)
    print(verdict)
    fig = plot_stage1(summary, run_dir / "stage1_results.png", synthetic=corpus.synthetic)
    print("figure:", fig)
    return summary


def go_no_go(summary, header_only: bool = False) -> str:
    """Stage-1 check from PLAN.md: the reproduced RNN must beat no-processing and MCTR, and stay below the IRM.
    header_only=True: just the table of overall averages (used in the stage-4 verdict)."""
    t = summary.groupby("variant")[["stoi_mean", "ta_db_mean", "speech_preserved_mean"]].mean()
    get = lambda v, m: float(t.loc[v, m]) if v in t.index else float("nan")
    checks = [
        ("C (RNN-3) STOI > A (no processing)", get("C", "stoi_mean") > get("A", "stoi_mean")),
        ("C (RNN-3) STOI > B (MCTR)", get("C", "stoi_mean") > get("B", "stoi_mean")),
        ("C (RNN-3) STOI < I (ideal mask)", get("C", "stoi_mean") < get("I", "stoi_mean")),
        ("C removes more transient than B (dB)", get("C", "ta_db_mean") > get("B", "ta_db_mean")),
    ]
    lines = ["Overall averages (all test conditions and STRs):" if header_only else
             "Stage-1 go/no-go (averaged over all test conditions and STRs):"]
    for v in t.index:
        lines.append(f"  {v} {VARIANTS[v][0]:14s} STOI {get(v, 'stoi_mean'):.3f}   transient attenuation "
                     f"{get(v, 'ta_db_mean'):5.1f} dB   speech preserved {get(v, 'speech_preserved_mean'):.3f}")
    if header_only:
        return "\n".join(lines) + "\n"
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    lines.append("  => GO to stage 2" if all(ok for _, ok in checks) else "  => NO-GO: fix the baseline before stage 2")
    return "\n".join(lines) + "\n"


def plot_stage1(summary, out_path: Path, synthetic: bool = False, variants=("A", "B", "C", "D", "I"),
                title: str = "Stage 1 — reproducing the Keshavarzi et al. baseline (mean ± 95% CI over test sentences)") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from money_plot import GRID, INK, INK_2, MUTED, STYLE, SURFACE, _style_axes
    style = dict(STYLE)
    style["D"] = dict(label="Same RNN, continuous memory (D)", short="RNN-stream (D)", color="#6250d6",
                      lw=2.0, marker="o", ls="-", z=5)
    style["C"] = dict(STYLE["C"], label="Keshavarzi et al. 2021 RNN, reproduced (C)")
    from money_plot import ABLATION_STYLE
    style.update(ABLATION_STYLE)          # G, F: greys + line style (colour-blind-safe next to C/E/H)
    conds = [c for c in ("seen", "unseen", "grandma") if c in set(summary["cond"])]
    titles = {"seen": "Transient types seen in training", "unseen": "Unseen transient types",
              "grandma": "Crumpling & crinkling (never trained on)"}
    metrics = [("stoi", "Intelligibility of output (STOI)"), ("ta_db", "Transient attenuation (dB)"),
               ("speech_preserved", "Speech preserved (STOI of masked speech)")]
    fig, axes = plt.subplots(len(metrics), len(conds), figsize=(4.6 * len(conds), 3.4 * len(metrics)),
                             facecolor=SURFACE, squeeze=False, sharex=True)
    for j, cond in enumerate(conds):
        for i, (m, ylabel) in enumerate(metrics):
            ax = axes[i, j]; _style_axes(ax)
            for v in variants:
                d = summary[(summary.variant == v) & (summary.cond == cond)].sort_values("str_db", ascending=False)
                if d.empty:
                    continue
                st = style[v]
                ax.errorbar(d.str_db, d[f"{m}_mean"], yerr=d[f"{m}_ci"], color=st["color"], lw=st["lw"], ls=st["ls"],
                            marker=st["marker"] or None, ms=6, mec=SURFACE, capsize=2, label=st["label"], zorder=st["z"])
            ax.set_xlim(max(summary.str_db) + 1.5, min(summary.str_db) - 1.5)
            if i == 0:
                ax.set_title(titles[cond], loc="left", fontsize=11.5, color=INK, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10, color=INK)
            if i == len(metrics) - 1:
                ax.set_xlabel("Speech-to-transient ratio (dB)  →  louder transient", fontsize=10, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=13, color=INK, fontweight="bold")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC SMOKE TEST — NOT RESULTS", ha="center", va="center", rotation=25,
                 fontsize=34, color="#b8322f", alpha=0.25, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


###########################################################
#                 STAGE 3: MEMORIES (go/no-go)            #
###########################################################

def stage3(smoke: bool = False):
    import json, pickle
    import local_memory as lm
    from dbn_pretrain import DBN_DIR, load_dbn
    from config import DATA
    dbn_dir = CFG.out_dir / "smoke" / "dbn" if smoke else DBN_DIR
    cache = DATA / "synthetic_cache" if smoke else CFG.data.cache_dir
    params, stats = load_dbn(dbn_dir)
    assert smoke or len(params) == 15, f"only {len(params)} of 15 LSNs are pretrained — finish dbn_pretrain.py first"
    DBN_DIR = dbn_dir
    memories = lm.build_memories(params, stats, cache, n_speech_crops=16 if smoke else 200)
    pickle.dump(memories, open(DBN_DIR / "memories.pkl", "wb"))
    rep = lm.separability_report(params, stats, memories, cache)
    (DBN_DIR / "memory_separability.json").write_text(json.dumps(rep, indent=1))
    lines = ["Stage-3 memory check (held-out speech vs held-out transients; speech share of the belief):"]
    for k, v in rep.items():
        lines.append(f"  {k}: on speech {v['speech_share_on_speech']:.2f}  on transients "
                     f"{v['speech_share_on_transient']:.2f}  gap {v['gap']:+.2f}")
    good = sum(v["gap"] > 0.2 for v in rep.values())
    lines.append(f"  {good}/{len(rep)} sub-networks separate speech from transients (gap > 0.2) -> "
                 f"{'GO' if good >= 3 else 'NO-GO: attention has little to work with'}")
    text = "\n".join(lines) + "\n"
    (DBN_DIR / "stage3_go_no_go.txt").write_text(text)
    print(text)
    return rep


###########################################################
#          STAGE 4: BELIEF-NETWORK MODELS + SCORING       #
###########################################################

ALL_VARIANTS = ("A", "B", "C", "D", "E", "F", "G", "H", "I")


def paired_comparison(df, a: str = "H", against=("C", "D", "E"), conds=("seen", "unseen", "grandma")) -> str:
    """Per-mixture paired differences (same test audio for every model) with 95% CI and Wilcoxon p."""
    from scipy.stats import wilcoxon
    lines = [f"Paired comparison of {a} vs baselines (difference = {a} minus baseline; same test mixtures):"]
    for b in against:
        for cond in conds:
            x = df[(df.variant == a) & (df.cond == cond)].sort_values(["str_db", "speech_idx"])
            y = df[(df.variant == b) & (df.cond == cond)].sort_values(["str_db", "speech_idx"])
            if x.empty or y.empty:
                continue
            parts = []
            for m in ("stoi", "ta_db", "speech_preserved"):
                d = x[m].to_numpy() - y[m].to_numpy()
                ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                parts.append(f"{m} {d.mean():+.4f} ± {ci:.4f} (p={p:.1e})")
            lines.append(f"  {a}-{b} [{cond:7s}] " + "   ".join(parts))
    return "\n".join(lines) + "\n"


ABLATIONS = (  # (model, against, the question the difference answers)
    ("H", "G", "does the attention gate help? (H minus G: same model, attention off)"),
    ("H", "F", "does keeping the filterbank help? (H minus F: belief network without the raw 64 bands)"),
    ("G", "E", "do belief-network features beat simply a bigger LSTM? (G minus E: no attention, size-matched)"),
)


def ablation_comparison(df) -> str:
    """Paired tests for the three design questions (same test mixtures for every model)."""
    lines = ["Ablations (paired, same test mixtures; difference = first model minus second):"]
    for a, b, question in ABLATIONS:
        lines.append(f" {question}")
        lines += paired_comparison(df, a, against=(b,)).splitlines()[1:]
    return "\n".join(lines) + "\n"


def equal_quality_table(to, levels=None):
    """Transient attenuation each model reaches when tuned (mask ** p) to the SAME speech-preserved level."""
    import pandas as pd
    from money_plot import _x_at
    if levels is None:
        levels = (float(to[(to.variant == "C") & (to.p == 1.0)].sp_mean.iloc[0]), 0.975, 0.965)
    rows = []
    for v in sorted(set(to.variant)):
        rows.append({"variant": v, **{f"ta_db_at_sp_{y:.3f}": _x_at(to[to.variant == v], y) for y in levels}})
    return pd.DataFrame(rows)


def tradeoff(corpus, rows, models, variants=("C", "D", "E", "F", "G", "H"), exponents=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
             cond="unseen", str_db=-15.0, workers=4):
    """Money-plot left panel: make each model more/less aggressive (mask ** p) -> attenuation vs speech kept."""
    import jax.numpy as jnp
    import frontend_gammatone as fe
    import pandas as pd
    rows = [r for r in rows if r["cond"] == cond and r["str_db"] == str_db]
    mixes = [dm.realise(corpus, r) for r in rows]
    out = []
    for v in variants:
        masks = [o["mask"] for o in mv.process_batch(v, mixes, models.get(v))]
        for p in exponents:
            items = []
            for m, M in zip(mixes, masks):
                Mp = jnp.asarray(M) ** p
                items.append(dict(s=m["s"], v=m["v"], onset=m["onset"],
                                  s_hat=np.asarray(fe.enhance_with_mask(jnp.asarray(m["s"]), Mp)),
                                  v_hat=np.asarray(fe.enhance_with_mask(jnp.asarray(m["v"]), Mp))))
            sc = em.score_many_tradeoff(items, workers)
            out.append(dict(variant=v, p=p, ta_db_mean=np.mean([r["ta_db"] for r in sc]),
                            sp_mean=np.mean([r["speech_preserved"] for r in sc]), n=len(sc)))
            print(f"  tradeoff {v} p={p}: TA {out[-1]['ta_db_mean']:.1f} dB, speech preserved {out[-1]['sp_mean']:.3f}", flush=True)
    return pd.DataFrame(out)


def stage4(n_per_cell: int = 100, steps: int = 2000, batch: int = 16, seed: int = 0, retrain: bool = False,
           variants_to_train=("H", "G", "F", "E"), smoke: bool = False):
    import pandas as pd
    if smoke:
        import synthetic_audio
        corpus = dm.load_corpus(synthetic_audio.build())
        mv.set_dbn_dir(CFG.out_dir / "smoke" / "dbn")
        steps, batch, n_per_cell = 20, 4, 4
    else:
        corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    recipes = tm.get_recipes(corpus, run_dir)
    models = {}
    for v in ("C", "D") + tuple(variants_to_train):
        ckpt = run_dir / v / f"seed{seed}" / "best.pkl"
        if retrain and v in variants_to_train or not ckpt.exists():
            tm.train(v, corpus, n_steps=steps, batch_size=batch, seed=seed, val_every=10 if smoke else 100)
        models[v] = mv.Trained.load(ckpt)
    rows = select_test_rows(recipes, n_per_cell)
    variants = [v for v in ALL_VARIANTS if v in models or v in ("A", "B", "I")]
    results = evaluate(corpus, run_dir, rows, models, variants=variants)
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "stage4_results_per_utterance.csv", index=False)
    summary = em.aggregate(results)
    summary.to_csv(run_dir / "stage4_summary.csv", index=False)
    verdict = go_no_go(summary, header_only=True) + "\n" + paired_comparison(df) + "\n" + ablation_comparison(df)
    (run_dir / "stage4_verdict.txt").write_text(verdict)
    print(verdict)
    plot_stage1(summary, run_dir / "stage4_results.png", variants=("A", "C", "E", "H", "I"),
                title="Stage 4 — belief network + attention (H) vs baselines (mean ± 95% CI over test sentences)")
    plot_stage1(summary, run_dir / "stage4_ablations.png", variants=("C", "F", "G", "H", "I"),
                title="Stage 4 ablations — what each part of the proposed model contributes")
    to = tradeoff(corpus, rows, models)
    to.to_csv(run_dir / "tradeoff_unseen_str-15.csv", index=False)
    eq = equal_quality_table(to)
    eq.to_csv(run_dir / "equal_speech_quality.csv", index=False)
    print("Transient attenuation (dB) at equal speech quality (unseen, STR -15 dB):\n" + eq.round(2).to_string(index=False))
    import money_plot
    print("money plot:", money_plot.make_money_plot(from_results=True, run_dir=run_dir))
    print("ablation frontier:", money_plot.make_frontier_plot(run_dir, run_dir / "stage4_ablation_frontier.png"))
    return summary


def load_models(run_dir: Path, variants, seed: int = 0) -> dict:
    return {v: mv.Trained.load(run_dir / v / f"seed{seed}" / "best.pkl") for v in variants
            if (run_dir / v / f"seed{seed}" / "best.pkl").exists()}


def refresh_stage4(variants=("F", "G"), n_per_cell: int = 100):
    """Add `variants` to the trade-off sweep (keeping rows already computed), then rewrite the verdict,
    the equal-quality table and the money plot from the saved per-utterance results. No retraining."""
    import pandas as pd
    import money_plot
    corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    rows = select_test_rows(tm.get_recipes(corpus, run_dir), n_per_cell)
    path = run_dir / "tradeoff_unseen_str-15.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["variant"])
    todo = [v for v in variants if v not in set(old.variant)]
    if todo:
        new = tradeoff(corpus, rows, load_models(run_dir, todo), variants=todo)
        old = pd.concat([old, new], ignore_index=True)
        old.to_csv(path, index=False)
    eq = equal_quality_table(old)
    eq.to_csv(run_dir / "equal_speech_quality.csv", index=False)
    print("Transient attenuation (dB) at equal speech quality (unseen, STR -15 dB):\n" + eq.round(2).to_string(index=False))
    df = pd.read_csv(run_dir / "stage4_results_per_utterance.csv")
    summary = pd.read_csv(run_dir / "stage4_summary.csv")
    verdict = go_no_go(summary, header_only=True) + "\n" + paired_comparison(df) + "\n" + ablation_comparison(df)
    (run_dir / "stage4_verdict.txt").write_text(verdict)
    print(verdict)
    print("money plot:", money_plot.make_money_plot(from_results=True, run_dir=run_dir))
    print("ablation frontier:", money_plot.make_frontier_plot(run_dir, run_dir / "stage4_ablation_frontier.png"))
    for name, vs, title in (
            ("stage4_results.png", ("A", "C", "E", "H", "I"),
             "Stage 4 — belief network + attention (H) vs baselines (mean ± 95% CI over test sentences)"),
            ("stage4_ablations.png", ("C", "F", "G", "H", "I"),
             "Stage 4 ablations — what each part of the proposed model contributes")):
        plot_stage1(summary, run_dir / name, variants=vs, title=title)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage3", "stage4", "refresh4"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="test sentences per (condition, STR) cell")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    if a.stage == "stage1":
        stage1(smoke=a.smoke, n_per_cell=a.n, steps=a.steps, batch=a.batch, retrain=a.retrain)
    elif a.stage == "stage3":
        stage3(smoke=a.smoke)
    elif a.stage == "refresh4":
        refresh_stage4(n_per_cell=a.n)
    else:
        stage4(n_per_cell=a.n, steps=a.steps, batch=a.batch, retrain=a.retrain or a.smoke, smoke=a.smoke)
