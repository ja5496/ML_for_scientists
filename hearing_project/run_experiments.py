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

USAGE (stage 1 = data + baseline reproduction, IMPLEMENTED)
------------------------------------------------------------
    python run_experiments.py stage1 --smoke          # 2-minute end-to-end check on synthetic data
    python run_experiments.py stage1                  # real data: trains C and D, scores A B C D I
    python run_experiments.py stage1 --n 300          # all 300 test sentences per cell (slower)
Outputs in runs/stage1/: results_per_utterance.csv, summary.csv, stage1_results.png, go_no_go.txt

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
        m.update(speech_idx=r["speech_idx"], trans=r["trans"], trans_idx=r["trans_idx"])
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


def go_no_go(summary) -> str:
    """Stage-1 check from PLAN.md: the reproduced RNN must beat no-processing and MCTR, and stay below the IRM."""
    t = summary.groupby("variant")[["stoi_mean", "ta_db_mean", "speech_preserved_mean"]].mean()
    get = lambda v, m: float(t.loc[v, m]) if v in t.index else float("nan")
    checks = [
        ("C (RNN-3) STOI > A (no processing)", get("C", "stoi_mean") > get("A", "stoi_mean")),
        ("C (RNN-3) STOI > B (MCTR)", get("C", "stoi_mean") > get("B", "stoi_mean")),
        ("C (RNN-3) STOI < I (ideal mask)", get("C", "stoi_mean") < get("I", "stoi_mean")),
        ("C removes more transient than B (dB)", get("C", "ta_db_mean") > get("B", "ta_db_mean")),
    ]
    lines = ["Stage-1 go/no-go (averaged over all test conditions and STRs):"]
    for v in t.index:
        lines.append(f"  {v} {VARIANTS[v][0]:12s} STOI {get(v, 'stoi_mean'):.3f}   transient attenuation "
                     f"{get(v, 'ta_db_mean'):5.1f} dB   speech preserved {get(v, 'speech_preserved_mean'):.3f}")
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    lines.append("  => GO to stage 2" if all(ok for _, ok in checks) else "  => NO-GO: fix the baseline before stage 2")
    return "\n".join(lines) + "\n"


def plot_stage1(summary, out_path: Path, synthetic: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from money_plot import GRID, INK, INK_2, MUTED, STYLE, SURFACE, _style_axes
    style = dict(STYLE)
    style["D"] = dict(label="Same RNN, continuous memory (D)", short="RNN-stream (D)", color="#6250d6",
                      lw=2.0, marker="o", ls="-", z=5)
    style["C"] = dict(STYLE["C"], label="Keshavarzi et al. 2021 RNN, reproduced (C)")
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
            for v in ("A", "B", "C", "D", "I"):
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
    title = "Stage 1 — reproducing the Keshavarzi et al. baseline (mean ± 95% CI over test sentences)"
    fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=13, color=INK, fontweight="bold")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC SMOKE TEST — NOT RESULTS", ha="center", va="center", rotation=25,
                 fontsize=34, color="#b8322f", alpha=0.25, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="test sentences per (condition, STR) cell")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    stage1(smoke=a.smoke, n_per_cell=a.n, steps=a.steps, batch=a.batch, retrain=a.retrain)
