# Hearing project: audible demo, proposed model (H) vs original RNN (C) - Jake Abraham
"""
PLAIN WORDS
-----------
Renders a few test mixtures through Keshavarzi's original RNN (C) and the proposed model (H) so you can hear
the difference, and writes a self-contained web page with a player that switches between versions while
keeping your place in the clip.

The examples are picked by a fixed rule from the scored test set, not by ear:
  1. a TYPICAL crinkling example (grandma's complaint): the mixture whose H-minus-C transient removal is the
     median over all 100 crinkling test sentences;
  2. a TYPICAL impact sound never heard in training (slam, knock, hammer, thump): again the median;
  3. H's BEST case on unseen transients (largest H-minus-C gain);
  4. H's WORST case (smallest H-minus-C gain, crinkling or unseen).
All at STR -15 dB (the transient's peak is 15 dB louder than the speech), the hardest level used in training.

Models: training seed 1, because seed 1 is the median seed for BOTH C and H on the equal-quality test (seed 0
was C's weakest seed, which would flatter H). Each example's clips share one gain, so relative loudness is kept.

Outputs (results/demo/): ex{k}_{clean,mix,C,H,ideal}.wav, ex{k}_spectrogram.png, demo.json, demo.html.

USAGE
-----
    python make_demo.py              # ~2 min CPU
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import data_mixing as dm
import model_variants as mv
import train_mask_estimator as tm
from config import CFG
from run_experiments import select_test_rows

SEED = 1
STR_DB = -15.0
OUT = Path(__file__).parent / "results" / "demo"
IMPACTS = {"Slam", "Knock", "Hammer", "Thump_and_thud"}
PRETTY = {"Crumpling_and_crinkling": "Crumpling and crinkling", "Slam": "Slam", "Knock": "Knock", "Hammer": "Hammer",
          "Thump_and_thud": "Thump and thud", "Coin_(dropping)": "Coin dropping",
          "Cutlery_and_silverware": "Cutlery and silverware"}
VERSIONS = [("clean", "Clean speech", "reference: no transient"),
            ("mix", "Unprocessed", "speech + transient"),
            ("C", "C · original RNN", "Keshavarzi et al. 2021"),
            ("H", "H · proposed", "filterbank + belief network + attention"),
            ("ideal", "Ideal mask", "ceiling: knows the answer")]


def pick_examples(run_dir: Path) -> list:
    """Rule-based choice of four mixtures from the seed-1 scores (see PLAIN WORDS)."""
    r = pd.read_csv(run_dir / "multiseed" / f"seed{SEED}_results.csv")
    r = r[r.str_db == STR_DB]
    key = ["cond", "str_db", "speech_idx"]
    c = r[r.variant == "C"].set_index(key)
    h = r[r.variant == "H"].set_index(key)
    d = h[["trans_label", "trans", "trans_idx", "talker"]].copy()
    d["gain"] = h.ta_db - c.ta_db
    d = d.reset_index()

    def median_row(sub):
        return sub.iloc[(sub.gain - sub.gain.median()).abs().argsort().iloc[0]]

    picks = [("Typical crinkling", "typical", median_row(d[d.cond == "grandma"])),
             ("Typical impact sound", "typical", median_row(d[(d.cond == "unseen") & d.trans_label.isin(IMPACTS)])),
             ("Best case for H", "best", d[d.cond == "unseen"].sort_values("gain").iloc[-1])]
    used = {int(p[2].speech_idx) for p in picks}
    worst = d[d.cond.isin(["unseen", "grandma"]) & ~d.speech_idx.isin(used)].sort_values("gain").iloc[0]
    picks.append(("Worst case for H", "worst", worst))
    return picks


def scores(run_dir: Path, cond: str, idx: int) -> dict:
    """Official per-mixture scores: C and H from seed 1, the unprocessed mix (A) and ideal mask (I) from stage 4."""
    r1 = pd.read_csv(run_dir / "multiseed" / f"seed{SEED}_results.csv")
    r0 = pd.read_csv(run_dir / "stage4_results_per_utterance.csv")
    out = {}
    for name, df, v in (("mix", r0, "A"), ("C", r1, "C"), ("H", r1, "H"), ("ideal", r0, "I")):
        row = df[(df.variant == v) & (df.cond == cond) & (df.str_db == STR_DB) & (df.speech_idx == idx)].iloc[0]
        out[name] = dict(ta_db=float(row.ta_db), speech_preserved=float(row.speech_preserved), stoi=float(row.stoi))
    return out


def spectrogram_png(sigs: dict, onset_s: float, path: Path) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    from money_plot import INK, INK_2, STYLE, SURFACE
    fs = CFG.audio.sample_rate
    rows = [("clean", "Clean speech", INK_2), ("mix", "Unprocessed", INK_2),
            ("C", "C · original RNN", STYLE["C"]["color"]), ("H", "H · proposed", STYLE["H"]["color"])]
    specs = {}
    for k, _, _ in rows:
        f, t, Z = stft(sigs[k], fs=fs, nperseg=512, noverlap=512 - 128)
        specs[k] = 20 * np.log10(np.abs(Z) + 1e-9)
    vmax = specs["mix"].max()
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 6.4), sharex=True, facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.12))
    for ax, (k, lab, col) in zip(axes, rows):
        ax.pcolormesh(t, f / 1000, specs[k], vmin=vmax - 75, vmax=vmax, cmap="gray_r", shading="auto", rasterized=True)
        ax.set_ylim(0, 8)
        ax.axvline(onset_s, color=col, lw=1.0, ls=(0, (3, 3)), alpha=0.9)
        ax.text(0.005, 0.93, lab, transform=ax.transAxes, va="top", fontsize=10, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=col, lw=1.4))
        ax.set_ylabel("kHz", fontsize=9, color=INK_2)
        ax.tick_params(labelsize=8.5, colors=INK_2)
        for s in ax.spines.values():
            s.set_color("#c9c8c3")
    axes[-1].set_xlabel("Time (s)   ·   dashed line = transient onset", fontsize=9.5, color=INK_2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def wav_bytes(x: np.ndarray, path: Path) -> bytes:
    fs = CFG.audio.sample_rate
    sf.write(path, x, fs, subtype="PCM_16")
    buf = io.BytesIO()
    sf.write(buf, x, fs, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = dm.load_corpus()
    run_dir = tm.run_dir_for(corpus)
    rows = select_test_rows(tm.get_recipes(corpus, run_dir), 100)
    models = {v: mv.Trained.load(run_dir / v / f"seed{SEED}" / "best.pkl") for v in ("C", "H")}
    fs = CFG.audio.sample_rate
    examples = []
    for k, (title, kind, p) in enumerate(pick_examples(run_dir), start=1):
        row = next(r for r in rows if r["cond"] == p.cond and r["str_db"] == STR_DB and r["speech_idx"] == p.speech_idx)
        mix = dm.realise(corpus, row)
        n = len(mix["x"])
        fit = lambda y: np.pad(np.asarray(y, np.float32)[:n], (0, max(0, n - len(y))))
        sigs = {"clean": fit(mix["s"]), "mix": fit(mix["x"]),
                "C": fit(mv.process_batch("C", [mix], models["C"])[0]["y"]),
                "H": fit(mv.process_batch("H", [mix], models["H"])[0]["y"]),
                "ideal": fit(mv.process_batch("I", [mix])[0]["y"])}
        g = 0.89 / max(np.abs(v).max() for v in sigs.values())          # one gain per example: levels stay comparable
        sigs = {kk: v * g for kk, v in sigs.items()}
        audio = {kk: base64.b64encode(wav_bytes(v, OUT / f"ex{k}_{kk}.wav")).decode() for kk, v in sigs.items()}
        png = base64.b64encode(spectrogram_png(sigs, mix["onset"] / fs, OUT / f"ex{k}_spectrogram.png")).decode()
        sc = scores(run_dir, p.cond, int(p.speech_idx))
        label = PRETTY.get(p.trans_label, str(p.trans_label).replace("_", " "))
        examples.append(dict(
            id=f"ex{k}", title=title, kind=kind, transient=label, talker=str(p.talker), str_db=STR_DB,
            heard_in_training=False, cond=p.cond, speech_idx=int(p.speech_idx), seconds=round(n / fs, 2),
            onset_s=round(mix["onset"] / fs, 2), peak_s=round((mix["onset"] + dm.PRE_PEAK) / fs, 2),
            gain_db=round(float(p.gain), 2), scores=sc,
            audio=audio, spectrogram=png))
        print(f"ex{k} {title}: {label}, talker {p.talker}, H-C transient removal {p.gain:+.1f} dB", flush=True)
    meta = [{k: v for k, v in e.items() if k not in ("audio", "spectrogram")} for e in examples]
    (OUT / "demo.json").write_text(json.dumps(dict(seed=SEED, str_db=STR_DB, examples=meta), indent=2))
    data = dict(versions=[dict(key=k, label=l, sub=s) for k, l, s in VERSIONS], examples=examples)
    page = (Path(__file__).parent / "demo_template.html").read_text().replace("__DATA__", json.dumps(data))
    (OUT / "demo.html").write_text(page)
    print("wrote", OUT / "demo.html", f"({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
