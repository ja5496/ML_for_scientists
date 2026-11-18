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
    python run_experiments.py multiseed               # seeds 1-2 for C, E, G, H: do the stage-4 gaps survive retraining?
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
        outs = mv.process_batch(models[v].variant if v in models else v, mixes, models.get(v))   # v may be a label like H_ft
        items = [dict(m, **o, variant=v) for m, o in zip(mixes, outs)]
        for it in items:
            it.pop("mask", None)
        print(f"  scoring {v} ({VARIANTS[v][0] if v in VARIANTS else v}) on {len(items)} mixtures ...", flush=True)
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
        masks = [o["mask"] for o in mv.process_batch(models[v].variant if v in models else v, mixes, models.get(v))]
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


def multiseed(seeds=(0, 1, 2), variants=("C", "E", "G", "H"), n_per_cell: int = 100, steps: int = 2000, batch: int = 16):
    """Retrain the key models with more seeds (init AND data order change) and ask whether the stage-4 gaps survive.
    Resumable: models are skipped if their checkpoint exists, and each seed's scores are cached in runs/stage1/multiseed/.
    Seed 0 reuses the stage-4 results. Writes multiseed_summary.txt, multiseed_*.csv and multiseed_seeds.png."""
    import pandas as pd
    corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    out = run_dir / "multiseed"; out.mkdir(exist_ok=True)
    rows = select_test_rows(tm.get_recipes(corpus, run_dir), n_per_cell)
    for seed in seeds:                                         # 1) train everything first (the slow, resumable part)
        for v in variants:
            if not (run_dir / v / f"seed{seed}" / "best.pkl").exists():
                tm.train(v, corpus, n_steps=steps, batch_size=batch, seed=seed)
    per_utt, trade = [], []
    for seed in seeds:                                         # 2) score (cached per seed)
        f_res, f_to = out / f"seed{seed}_results.csv", out / f"seed{seed}_tradeoff.csv"
        if seed == 0 and not f_res.exists():
            df0 = pd.read_csv(run_dir / "stage4_results_per_utterance.csv")
            df0[df0.variant.isin(variants)].to_csv(f_res, index=False)
            to0 = pd.read_csv(run_dir / "tradeoff_unseen_str-15.csv")
            to0[to0.variant.isin(variants)].to_csv(f_to, index=False)
        if not f_res.exists():
            pd.DataFrame(evaluate(corpus, run_dir, rows, load_models(run_dir, variants, seed), variants=variants)).to_csv(f_res, index=False)
        if not f_to.exists():
            tradeoff(corpus, rows, load_models(run_dir, variants, seed), variants=variants).to_csv(f_to, index=False)
        per_utt.append(pd.read_csv(f_res).assign(seed=seed))
        trade.append(pd.read_csv(f_to).assign(seed=seed))
    df, to = pd.concat(per_utt, ignore_index=True), pd.concat(trade, ignore_index=True)

    # 3) per-seed paired differences (same mixtures), then mean ± SD over seeds
    pairs = [("H", "E"), ("H", "C"), ("H", "G"), ("G", "E")]
    key = ["cond", "str_db", "speech_idx"]
    recs = []
    for seed in seeds:
        d = df[df.seed == seed]
        for a, b in pairs:
            x = d[d.variant == a].set_index(key).sort_index(); y = d[d.variant == b].set_index(key).sort_index()
            for cond in ("seen", "unseen", "grandma"):
                for m in ("stoi", "ta_db", "speech_preserved"):
                    recs.append(dict(seed=seed, pair=f"{a}-{b}", cond=cond, metric=m,
                                     diff=float((x.loc[cond, m] - y.loc[cond, m]).mean())))
    diffs = pd.DataFrame(recs)
    diffs.to_csv(run_dir / "multiseed_paired_diffs.csv", index=False)
    eq = pd.concat([equal_quality_table(to[to.seed == s], levels=(0.985, 0.975, 0.965)).assign(seed=s) for s in seeds])
    eq.to_csv(run_dir / "multiseed_equal_quality.csv", index=False)

    # 4) seed-averaged predictions: average each mixture's score over seeds, then one paired test per pair
    from scipy.stats import wilcoxon
    avg = df.groupby(["variant"] + key)[["stoi", "ta_db", "speech_preserved"]].mean()
    lines = [f"Multi-seed check: seeds {list(seeds)}, models {list(variants)}, {n_per_cell} sentences per cell.",
             "", "A) Per-seed paired difference (first minus second, mean over the cell's mixtures); mean ± SD over seeds:"]
    for (pair, cond, m), g in diffs.groupby(["pair", "cond", "metric"], sort=False):
        vals = g.sort_values("seed")["diff"].to_numpy()
        lines.append(f"  {pair} {cond:7s} {m:16s} " + "  ".join(f"{v:+.4f}" for v in vals)
                     + f"   mean {vals.mean():+.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}"
                     + f"   ({int((vals > 0).sum())}/{len(vals)} seeds > 0)")
    lines += ["", "B) Transient attenuation (dB) at equal speech quality, unseen types, STR -15 dB, per seed:"]
    for v in variants:
        g = eq[eq.variant == v].sort_values("seed")
        cols = [c for c in eq.columns if c.startswith("ta_db_at")]
        lines.append(f"  {v}  " + "   ".join(f"{c[-5:]}: " + " ".join(f"{x:5.2f}" for x in g[c]) + f" (mean {g[c].mean():5.2f})"
                                          for c in cols))
    lines += ["", "C) Seed-averaged models (each mixture's score averaged over seeds), paired Wilcoxon over mixtures:"]
    for a, b in pairs:
        for cond in ("seen", "unseen", "grandma"):
            parts = []
            for m in ("stoi", "ta_db", "speech_preserved"):
                d = (avg.loc[a].loc[cond][m] - avg.loc[b].loc[cond][m]).to_numpy()
                parts.append(f"{m} {d.mean():+.4f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.4f} (p={wilcoxon(d).pvalue:.1e})")
            lines.append(f"  {a}-{b} [{cond:7s}] " + "   ".join(parts))
    text = "\n".join(lines) + "\n"
    (run_dir / "multiseed_summary.txt").write_text(text)
    print(text)
    _plot_multiseed(eq, run_dir / "multiseed_seeds.png", variants)
    import money_plot                                          # headline figures now average over the seeds
    print("money plot:", money_plot.make_money_plot(from_results=True, run_dir=run_dir))
    print("ablation frontier:", money_plot.make_frontier_plot(run_dir, run_dir / "stage4_ablation_frontier.png"))
    return text


def _plot_multiseed(eq, out_path: Path, variants, style=None, title="Does the ranking survive retraining?"):
    """One dot per seed: transient removed at speech preserved 0.975 (unseen, -15 dB). Spread = seed noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from money_plot import ABLATION_STYLE, INK, INK_2, STYLE, SURFACE, _style_axes
    style = style or {**STYLE, **ABLATION_STYLE}
    col = "ta_db_at_sp_0.975"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    seeds = sorted(eq.seed.unique())
    marks, offs = ("o", "s", "^", "D", "v"), np.linspace(0.16, -0.16, len(seeds)) if len(seeds) > 1 else [0.0]
    for i, v in enumerate(variants):
        g = eq[eq.variant == v]
        for k, s in enumerate(seeds):           # each seed on its own sub-row + shape, so equal values never hide
            x = g[g.seed == s][col].to_numpy()
            ax.scatter(x, np.full(len(x), i + offs[k]), s=64, marker=marks[k % len(marks)], color=style[v]["color"],
                       edgecolor=SURFACE, linewidth=1.2, zorder=3, label=f"seed {s}" if i == 0 else None)
        ax.plot([g[col].mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4, label="mean" if i == 0 else None)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK, ncol=len(seeds) + 1)
    for h in leg.legend_handles[:-1]:
        h.set_color(INK_2)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([style[v]["label"] if f"({v})" in style[v]["label"] else f"{style[v]['label']} ({v})"
                        for v in variants], fontsize=9.5, color=INK)
    ax.set_xlabel("Transient attenuation (dB) at speech preserved = 0.975   →  better", fontsize=10.5, color=INK)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=26)
    ax.text(0, 1.02, "Unseen transient types, STR −15 dB. Marker = one training seed (shape = which seed); bar = mean over seeds.",
            transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")
    ax.set_ylim(-0.95, len(variants) - 0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


###########################################################
#          STAGE 5: END-TO-END FINE-TUNING (H_ft)         #
###########################################################

FT_ARMS = (("H", True), ("H", False), ("E", False))        # -> H_ft, H_cont, E_cont (all: +2,000 steps on new data)
FT_LABELS = ("H_ft", "H_cont", "E_cont")


def stage5(seeds=(0, 1, 2), n_per_cell: int = 100, steps: int = 2000, batch: int = 16, smoke: bool = False):
    """Fine-tune H end to end (belief network + attention unfrozen) and compare with two controls that get exactly
    the same extra training with everything else frozen: H_cont (same model) and E_cont (the size-matched LSTM).
    Resumable: a run is skipped once its `done` file exists; each seed's scores are cached in runs/stage1/stage5/.
    The summary is rewritten after every seed, so partial results are readable while later seeds train."""
    import pandas as pd
    import traceback
    if smoke:                                               # needs `stage4 --smoke` first
        import synthetic_audio
        corpus = dm.load_corpus(synthetic_audio.build())
        mv.set_dbn_dir(CFG.out_dir / "smoke" / "dbn")
        seeds, steps, batch, n_per_cell = (0,), 6, 4, 4
    else:
        corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    out = run_dir / "stage5"; out.mkdir(exist_ok=True)
    rows = select_test_rows(tm.get_recipes(corpus, run_dir), n_per_cell)
    for seed in seeds:
        for (v, unfreeze), lab in zip(FT_ARMS, FT_LABELS):
            if not (run_dir / lab / f"seed{seed}" / "done").exists():
                tm.finetune(v, corpus, seed=seed, n_steps=steps, batch_size=batch, unfreeze=unfreeze,
                            val_every=3 if smoke else 100)
        models = {lab: mv.Trained.load(run_dir / lab / f"seed{seed}" / "best.pkl") for lab in FT_LABELS}
        f_res, f_to = out / f"seed{seed}_results.csv", out / f"seed{seed}_tradeoff.csv"
        if not f_res.exists():
            pd.DataFrame(evaluate(corpus, run_dir, rows, models, variants=FT_LABELS)).to_csv(f_res, index=False)
        if not f_to.exists():
            tradeoff(corpus, rows, models, variants=FT_LABELS).to_csv(f_to, index=False)
        try:                                                # a plotting bug must not stop the next seed's training
            print(summarise_stage5(run_dir), flush=True)
        except Exception:
            traceback.print_exc()


def _stage5_tables(run_dir: Path):
    """Per-mixture results and trade-off curves for C, E, H (2,000 steps) and H_ft, H_cont, E_cont, every seed
    that has stage-5 scores."""
    import pandas as pd
    out = run_dir / "stage5"
    seeds = sorted(int(f.name[4:].split("_")[0]) for f in out.glob("seed*_tradeoff.csv"))
    res, to = [], []
    for k in seeds:
        if k == 0:
            r0 = pd.read_csv(run_dir / "stage4_results_per_utterance.csv"); t0 = pd.read_csv(run_dir / "tradeoff_unseen_str-15.csv")
        else:
            r0 = pd.read_csv(run_dir / "multiseed" / f"seed{k}_results.csv"); t0 = pd.read_csv(run_dir / "multiseed" / f"seed{k}_tradeoff.csv")
        keep = ["C", "E", "H"]
        res.append(pd.concat([r0[r0.variant.isin(keep)], pd.read_csv(out / f"seed{k}_results.csv")]).assign(seed=k))
        to.append(pd.concat([t0[t0.variant.isin(keep)], pd.read_csv(out / f"seed{k}_tradeoff.csv")]).assign(seed=k))
    return seeds, pd.concat(res, ignore_index=True), pd.concat(to, ignore_index=True)


def summarise_stage5(run_dir: Path) -> str:
    import pandas as pd
    from scipy.stats import wilcoxon
    seeds, df, to = _stage5_tables(run_dir)
    pairs = [("H_ft", "H_cont"), ("H_cont", "H"), ("H_ft", "E_cont"), ("H_cont", "E_cont"), ("E_cont", "E")]
    key = ["cond", "str_db", "speech_idx"]
    recs = []
    for seed in seeds:
        d = df[df.seed == seed]
        for a, b in pairs:
            x = d[d.variant == a].set_index(key).sort_index(); y = d[d.variant == b].set_index(key).sort_index()
            for cond in ("seen", "unseen", "grandma"):
                for m in ("stoi", "ta_db", "speech_preserved"):
                    recs.append(dict(seed=seed, pair=f"{a}-{b}", cond=cond, metric=m,
                                     diff=float((x.loc[cond, m] - y.loc[cond, m]).mean())))
    diffs = pd.DataFrame(recs)
    diffs.to_csv(run_dir / "stage5_paired_diffs.csv", index=False)
    variants = ["C", "E", "E_cont", "H", "H_cont", "H_ft"]
    eq = pd.concat([equal_quality_table(to[(to.seed == s) & to.variant.isin(variants)], levels=(0.985, 0.975, 0.965)).assign(seed=s)
                    for s in seeds])
    eq.to_csv(run_dir / "stage5_equal_quality.csv", index=False)

    lines = [f"Stage 5 (end-to-end fine-tuning), seeds {seeds}. H_ft, H_cont, E_cont = +2,000 steps on new data from the",
             "stage-4 checkpoints (H_ft: belief network + attention unfrozen; H_cont / E_cont: frozen controls).", ""]
    lines.append("Validation mask error (MSE): start -> best, and the attention settings H_ft learned:")
    for seed in seeds:
        parts = []
        for lab in FT_LABELS:
            lg = pd.read_csv(run_dir / lab / f"seed{seed}" / "log.csv")
            parts.append(f"{lab} {lg.val_loss.iloc[0]:.5f} -> {lg.val_loss.min():.5f} (step {int(lg.step[lg.val_loss.idxmin()])})")
        m = mv.Trained.load(run_dir / "H_ft" / f"seed{seed}" / "best.pkl")
        a = mv.attn_settings(m.extra["attn"]) if m.extra and "attn" in m.extra else None
        att = (f"   alpha {float(a['alpha']):.3f} beta {float(a['beta']):+.3f} T {float(a['temperature']):.4f}"
               f" (start 0.700 / -0.100 / 0.0500)") if a else ""
        lines.append(f"  seed {seed}: " + "   ".join(parts) + att)
    lines += ["", "A) Per-seed paired difference (first minus second); mean ± SD over seeds:"]
    for (pair, cond, m), g in diffs.groupby(["pair", "cond", "metric"], sort=False):
        vals = g.sort_values("seed")["diff"].to_numpy()
        lines.append(f"  {pair:14s} {cond:7s} {m:16s} " + "  ".join(f"{v:+.4f}" for v in vals)
                     + f"   mean {vals.mean():+.4f}" + (f" ± {vals.std(ddof=1):.4f}" if len(vals) > 1 else "")
                     + f"   ({int((vals > 0).sum())}/{len(vals)} seeds > 0)")
    lines += ["", "B) Transient attenuation (dB) at equal speech quality, unseen types, STR -15 dB, per seed:"]
    cols = [c for c in eq.columns if c.startswith("ta_db_at")]
    for v in variants:
        g = eq[eq.variant == v].sort_values("seed")
        lines.append(f"  {v:7s} " + "   ".join(f"{c[-5:]}: " + " ".join(f"{x:5.2f}" for x in g[c]) + f" (mean {g[c].mean():5.2f})"
                                            for c in cols))
    if len(seeds) > 0:
        avg = df.groupby(["variant"] + key)[["stoi", "ta_db", "speech_preserved"]].mean()
        lines += ["", "C) Seed-averaged models, paired Wilcoxon over mixtures (test-set noise only, not training noise):"]
        for a, b in pairs:
            for cond in ("seen", "unseen", "grandma"):
                parts = []
                for m in ("stoi", "ta_db", "speech_preserved"):
                    d = (avg.loc[a].loc[cond][m] - avg.loc[b].loc[cond][m]).to_numpy()
                    parts.append(f"{m} {d.mean():+.4f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.4f} (p={wilcoxon(d).pvalue:.1e})")
                lines.append(f"  {a}-{b} [{cond:7s}] " + "   ".join(parts))
    text = "\n".join(lines) + "\n"
    (run_dir / "stage5_summary.txt").write_text(text)
    style = _stage5_style()
    _plot_multiseed(eq, run_dir / "stage5_seeds.png", ["E", "E_cont", "H", "H_cont", "H_ft"], style=style,
                    title="Does end-to-end fine-tuning help?")
    _plot_stage5_frontier(to, run_dir / "stage5_frontier.png", style)
    return text


def _stage5_style():
    from money_plot import ABLATION_STYLE, MUTED, STYLE
    blue, green = STYLE["H"]["color"], STYLE["E"]["color"]
    return {**STYLE, **ABLATION_STYLE,
            # the 2,000-step starting points keep their model's hue (colour follows the entity), as a pale dotted line
            "H": dict(STYLE["H"], label="Proposed, 2,000 steps (H)", lw=1.6, ls=":", color="#86b4ea", marker="o", z=2),
            "E": dict(STYLE["E"], label="Size-matched LSTM, 2,000 steps (E)", lw=1.6, ls=":", color="#7ccfae", marker="s",
                      z=2),
            "H_ft": dict(label="Proposed, +2,000 steps, belief network + attention fine-tuned (H_ft)", short="Fine-tuned (H_ft)",
                         color=blue, lw=3.0, marker="o", ls="-", z=6),
            "H_cont": dict(label="Proposed, +2,000 steps, belief network frozen (H_cont)", short="Frozen (H_cont)",
                           color=blue, lw=1.8, marker="o", ls="--", z=5),
            "E_cont": dict(label="Size-matched LSTM, +2,000 steps (E_cont)", short="LSTM (E_cont)",
                           color=green, lw=2.0, marker="s", ls="-", z=4)}


def _plot_stage5_frontier(to, out_path: Path, style):
    """Seed-averaged trade-off curves: the three equal-budget models plus H and E at 2,000 steps for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from money_plot import INK, INK_2, SURFACE, _mean_curves, _style_axes, _x_at
    mean = _mean_curves(to)
    plotted = []
    n = to.seed.nunique()
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor=SURFACE)
    _style_axes(ax)
    for v in ("H_ft", "H_cont", "E_cont", "H", "E"):
        c = mean[mean.variant == v].sort_values("p")
        if c.empty:
            continue
        st = style[v]
        plotted.append(v)
        ax.plot(c.ta_db_mean, c.sp_mean, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st.get("z", 3),
                label=st["label"], marker=".", ms=5)
        op = c[c.p == 1.0]
        ax.plot(op.ta_db_mean, op.sp_mean, marker=st["marker"], ms=9, color=st["color"], mec=SURFACE, mew=2,
                zorder=st.get("z", 3) + 1, ls="none")
    ax.set_xlabel("Transient attenuation (dB)   →  more comfortable", fontsize=11, color=INK)
    ax.set_ylabel("Speech preserved (STOI of masked clean speech)", fontsize=11, color=INK)
    ax.set_title("Stage 5: end-to-end fine-tuning vs equal extra training", loc="left", fontsize=12.5, color=INK,
                 fontweight="bold", pad=34)
    ax.text(0, 1.02, f"Unseen transient types, STR −15 dB, mean of {n} training seed{'s' if n > 1 else ''}. "
            "Big marker = model as trained; line = mask ** p.\nFurther right at the same height = more transient removed "
            "for the same speech damage.", transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")
    # legend best-first: ordered by transient removed where each curve crosses speech preserved = 0.97
    h, lab = ax.get_legend_handles_labels()
    x97 = [_x_at(mean[mean.variant == v], 0.97) for v in plotted]
    order = sorted(range(len(h)), key=lambda i: -np.nan_to_num(x97[i], nan=-1e9))
    ax.legend([h[i] for i in order], [lab[i] for i in order], loc="lower left", frameon=False, fontsize=9,
              labelcolor=INK, title="Best first (most removed at speech preserved 0.97)", title_fontsize=8.5,
              alignment="left")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


###########################################################
#       STAGE 6: LONGER TRAINING (does H's lead hold?)    #
###########################################################

LONG_MODELS = ("C", "E", "H")
BUDGETS = (2000, 4000, 8000)


def stage6(seeds=(0, 1, 2), n_per_cell: int = 100, long_steps: int = 4000, batch: int = 16, smoke: bool = False):
    """Train C, E and H for longer and track how far apart they stay. Stage 5 found that 2,000 extra steps are worth
    about as much as H's lead, and that the lead shrank a little, so the headline needs checking toward convergence.

    Training budgets, with every model on the same data stream within a seed:
      2,000 steps  stage 4: C, E, H
      4,000 steps  stage 5 controls E_cont, H_cont, plus C_cont trained here on the same stage-5 data (data seed + 100)
      8,000 steps  C_long, E_long, H_long: +4,000 steps on new data (data seed + 200), from the 4,000-step checkpoints
    All frozen-belief-network runs (no fine-tuning). Resumable: runs with a `done` file are skipped and each seed's scores
    are cached in runs/stage1/stage6/. The summary and learning-curve figure are rewritten after every seed."""
    import pandas as pd
    import traceback
    if smoke:                                               # needs `stage4 --smoke` and `stage5 --smoke` first
        import synthetic_audio
        corpus = dm.load_corpus(synthetic_audio.build())
        mv.set_dbn_dir(CFG.out_dir / "smoke" / "dbn")
        seeds, batch, n_per_cell = (0,), 4, 4
    else:
        corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    out = run_dir / "stage6"; out.mkdir(exist_ok=True)
    rows = select_test_rows(tm.get_recipes(corpus, run_dir), n_per_cell)
    ve = 3 if smoke else 100
    labels = ["C_cont"] + [f"{v}_long" for v in LONG_MODELS]
    for seed in seeds:
        if not (run_dir / "C_cont" / f"seed{seed}" / "done").exists():
            tm.finetune("C", corpus, seed=seed, n_steps=6 if smoke else 2000, batch_size=batch, unfreeze=False, val_every=ve)
        for v in LONG_MODELS:
            if not (run_dir / f"{v}_long" / f"seed{seed}" / "done").exists():
                tm.finetune(v, corpus, seed=seed, n_steps=6 if smoke else long_steps, batch_size=batch, unfreeze=False,
                            data_seed_offset=200, start_label=f"{v}_cont", label=f"{v}_long", val_every=ve)
        models = {lab: mv.Trained.load(run_dir / lab / f"seed{seed}" / "best.pkl") for lab in labels}
        f_res, f_to = out / f"seed{seed}_results.csv", out / f"seed{seed}_tradeoff.csv"
        if not f_res.exists():
            pd.DataFrame(evaluate(corpus, run_dir, rows, models, variants=labels)).to_csv(f_res, index=False)
        if not f_to.exists():
            tradeoff(corpus, rows, models, variants=labels).to_csv(f_to, index=False)
        try:
            print(summarise_stage6(run_dir), flush=True)
        except Exception:
            traceback.print_exc()


def _stage6_tables(run_dir: Path):
    """Per-mixture results and trade-off curves for C, E, H at every training budget, for every seed with stage-6
    scores. Adds columns `model` (C/E/H) and `steps` (2,000 / 4,000 / 8,000)."""
    import pandas as pd
    out6, out5 = run_dir / "stage6", run_dir / "stage5"
    seeds = sorted(int(f.name[4:].split("_")[0]) for f in out6.glob("seed*_tradeoff.csv"))
    name = {"C": ("C", 2000), "E": ("E", 2000), "H": ("H", 2000), "C_cont": ("C", 4000), "E_cont": ("E", 4000),
            "H_cont": ("H", 4000), "C_long": ("C", 8000), "E_long": ("E", 8000), "H_long": ("H", 8000)}
    res, to = [], []
    for k in seeds:
        if k == 0:
            r2, t2 = run_dir / "stage4_results_per_utterance.csv", run_dir / "tradeoff_unseen_str-15.csv"
        else:
            r2, t2 = run_dir / "multiseed" / f"seed{k}_results.csv", run_dir / "multiseed" / f"seed{k}_tradeoff.csv"
        for f_r, f_t in ((r2, t2), (out5 / f"seed{k}_results.csv", out5 / f"seed{k}_tradeoff.csv"),
                         (out6 / f"seed{k}_results.csv", out6 / f"seed{k}_tradeoff.csv")):
            r, t = pd.read_csv(f_r), pd.read_csv(f_t)
            res.append(r[r.variant.isin(name)].assign(seed=k)); to.append(t[t.variant.isin(name)].assign(seed=k))
    res, to = pd.concat(res, ignore_index=True), pd.concat(to, ignore_index=True)
    for d in (res, to):
        d["model"] = d.variant.map(lambda v: name[v][0]); d["steps"] = d.variant.map(lambda v: name[v][1])
    return seeds, res, to


def summarise_stage6(run_dir: Path) -> str:
    import pandas as pd
    seeds, df, to = _stage6_tables(run_dir)
    levels = (0.985, 0.975, 0.965)
    eq = pd.concat([equal_quality_table(to[(to.seed == s) & (to.steps == n)], levels=levels).assign(seed=s, steps=n)
                    for s in seeds for n in BUDGETS])
    eq["model"] = eq.variant.str[0]
    eq.to_csv(run_dir / "stage6_equal_quality.csv", index=False)
    cols = [f"ta_db_at_sp_{y:.3f}" for y in levels]
    key = ["cond", "str_db", "speech_idx"]
    recs = []
    for s in seeds:
        for n in BUDGETS:
            d = df[(df.seed == s) & (df.steps == n)]
            m = {v: d[d.model == v].set_index(key).sort_index() for v in LONG_MODELS}
            for a, b in (("H", "E"), ("H", "C"), ("E", "C")):
                for cond in ("seen", "unseen", "grandma"):
                    for met in ("stoi", "ta_db", "speech_preserved"):
                        recs.append(dict(seed=s, steps=n, pair=f"{a}-{b}", cond=cond, metric=met,
                                         diff=float((m[a].loc[cond, met] - m[b].loc[cond, met]).mean())))
    diffs = pd.DataFrame(recs)
    diffs.to_csv(run_dir / "stage6_paired_diffs.csv", index=False)

    lines = [f"Stage 6 (longer training), seeds {seeds}. Budgets: 2,000 steps (stage 4), 4,000 (stage-5 controls + C_cont),",
             "8,000 (C_long, E_long, H_long). Belief network frozen throughout.", "",
             "Validation mask error (MSE) at the best checkpoint of each budget:"]
    for s in seeds:
        parts = []
        for v in LONG_MODELS:
            vals = []
            for lab in (v, f"{v}_cont", f"{v}_long"):
                lg = pd.read_csv(run_dir / lab / f"seed{s}" / "log.csv")
                vals.append(f"{lg.val_loss.min():.5f}")
            parts.append(f"{v} " + " -> ".join(vals))
        lines.append(f"  seed {s}: " + "   ".join(parts))
    lines += ["", "A) Transient attenuation (dB) at equal speech quality, unseen types, STR -15 dB (per seed, then mean):"]
    for v in LONG_MODELS:
        for n in BUDGETS:
            g = eq[(eq.model == v) & (eq.steps == n)].sort_values("seed")
            lines.append(f"  {v} {n:5d} steps  " + "   ".join(
                f"{c[-5:]}: " + " ".join(f"{x:5.2f}" for x in g[c]) + f" (mean {g[c].mean():5.2f})" for c in cols))
    lines += ["", "B) Gaps at equal speech quality (first minus second), per seed then mean:"]
    for a, b in (("H", "E"), ("H", "C")):
        for n in BUDGETS:
            ga = eq[(eq.model == a) & (eq.steps == n)].set_index("seed")[cols]
            gb = eq[(eq.model == b) & (eq.steps == n)].set_index("seed")[cols]
            dd = ga - gb
            lines.append(f"  {a}-{b} {n:5d} steps  " + "   ".join(
                f"{c[-5:]}: " + " ".join(f"{x:+5.2f}" for x in dd[c]) + f" (mean {dd[c].mean():+5.2f}, "
                f"{int((dd[c] > 0).sum())}/{len(dd)} > 0)" for c in cols))
    lines += ["", "C) As trained: paired difference over all 6 STRs, per seed then mean:"]
    for (pair, cond, met), g in diffs.groupby(["pair", "cond", "metric"], sort=False):
        for n in BUDGETS:
            vals = g[g.steps == n].sort_values("seed")["diff"].to_numpy()
            lines.append(f"  {pair} {n:5d} {cond:7s} {met:16s} " + " ".join(f"{x:+.4f}" for x in vals)
                         + f"   mean {vals.mean():+.4f}   ({int((vals > 0).sum())}/{len(vals)} > 0)")
    text = "\n".join(lines) + "\n"
    (run_dir / "stage6_summary.txt").write_text(text)
    _plot_learning_curve(eq, run_dir / "stage6_learning_curve.png")
    return text


def _label_line_ends(ax, ends, color, min_gap_frac: float = 0.06):
    """Direct labels at the right end of each line, nudged apart so close lines never get overlapping labels."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = max(0.0, ys[-1] - hi + gap / 2)
    for (x, y, lab), yl in zip(ends, ys):
        ax.annotate(lab, (x, y), xytext=(x * 1.08, yl - shift), textcoords="data", va="center", fontsize=9.5, color=color)


def _plot_learning_curve(eq, out_path: Path, col: str = "ta_db_at_sp_0.975"):
    """Left: transient removed at equal speech quality vs training steps, per model. Right: H's lead over E and C."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from money_plot import INK, INK_2, MUTED, STYLE, SURFACE, _style_axes
    seeds = sorted(eq.seed.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE, gridspec_kw=dict(wspace=0.32))
    for ax in (a1, a2):
        _style_axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS); ax.set_xticklabels([f"{n:,}" for n in BUDGETS])
        ax.minorticks_off()
        ax.set_xlim(BUDGETS[0] / 1.25, BUDGETS[-1] * 1.6)
        ax.set_xlabel("Training steps (batch 16, log scale)", fontsize=10.5, color=INK)
    names = {"H": "Proposed (H)", "E": "Size-matched LSTM (E)", "C": "Original RNN (C)"}
    ends1, ends2 = [], []
    for v in ("H", "E", "C"):
        g = eq[eq.model == v]
        col_v = STYLE[v]["color"]
        for s in seeds:
            gs = g[g.seed == s].sort_values("steps")
            a1.plot(gs.steps, gs[col], color=col_v, lw=1.0, alpha=0.35, zorder=2)
        m = g.groupby("steps")[col].mean()
        if len(m):
            a1.plot(m.index, m.values, color=col_v, lw=2.6, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=4, label=names[v])
            ends1.append((m.index[-1], m.values[-1], names[v]))
    a1.set_ylabel("Transient removed (dB) at speech preserved 0.975", fontsize=10.5, color=INK)
    a1.set_title("Transient removed vs training length", loc="left", fontsize=11.5, color=INK, fontweight="bold")
    gap_style = {"E": ("H − size-matched LSTM (E)", STYLE["E"]["color"]), "C": ("H − original RNN (C)", STYLE["C"]["color"])}
    a2.axhline(0, color=MUTED, lw=1.0, zorder=1)
    for b in ("E", "C"):
        lab, c = gap_style[b]
        piv = eq.pivot_table(index=["seed", "steps"], columns="model", values=col)
        if "H" not in piv or b not in piv:
            continue
        gap = (piv["H"] - piv[b]).reset_index(name="gap")
        for s in seeds:
            gs = gap[gap.seed == s].sort_values("steps")
            a2.plot(gs.steps, gs.gap, color=c, lw=1.0, alpha=0.35, zorder=2)
        m = gap.groupby("steps").gap.mean()
        a2.plot(m.index, m.values, color=c, lw=2.6, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=4, label=lab)
        ends2.append((m.index[-1], m.values[-1], lab))
    a2.set_ylabel("H's lead (dB) at speech preserved 0.975", fontsize=10.5, color=INK)
    a2.set_title("The proposed model's lead vs training length", loc="left", fontsize=11.5, color=INK, fontweight="bold")
    for ax, ends in ((a1, ends1), (a2, ends2)):
        _label_line_ends(ax, ends, INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(ends), frameon=False, fontsize=9,
                  labelcolor=INK, handlelength=2.2)
    fig.suptitle(f"Longer training · unseen transient types, STR −15 dB · thick = mean of {len(seeds)} seed"
                 f"{'s' if len(seeds) > 1 else ''}, thin = each seed", x=0.065, ha="left", y=1.02, fontsize=10, color=INK_2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage3", "stage4", "refresh4", "multiseed", "stage5", "summary5", "stage6", "summary6"])
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
    elif a.stage == "multiseed":
        multiseed(n_per_cell=a.n, steps=a.steps, batch=a.batch)
    elif a.stage == "stage5":
        stage5(n_per_cell=a.n, steps=a.steps, batch=a.batch, smoke=a.smoke)
    elif a.stage == "stage6":
        stage6(n_per_cell=a.n, batch=a.batch, smoke=a.smoke)
    elif a.stage == "summary6":
        print(summarise_stage6(CFG.out_dir / ("smoke" if a.smoke else "stage1")))
    elif a.stage == "summary5":
        print(summarise_stage5(CFG.out_dir / ("smoke" if a.smoke else "stage1")))
    else:
        stage4(n_per_cell=a.n, steps=a.steps, batch=a.batch, retrain=a.retrain or a.smoke, smoke=a.smoke)
