"""Two demo clips for the presentation: unprocessed -> original RNN (C) -> proposed (H), with a moving playhead
over stacked spectrograms. Picks are H's large wins over C at STR -15 dB (training seed 1), chosen by rule:
largest H-minus-C transient removal with speech kept within 0.02 and STOI not worse (see notes in the deck)."""
import subprocess
import sys
from pathlib import Path

import numpy as np

HP = Path("/home/user/ML_for_scientists/hearing_project")
sys.path.insert(0, str(HP))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

import data_mixing as dm
import model_variants as mv
import train_mask_estimator as tm
from config import CFG
from run_experiments import select_test_rows

OUT = Path(__file__).parent / "clips"
OUT.mkdir(exist_ok=True)
SEED, STR_DB, FPS, W, H = 1, -15.0, 25, 1280, 720
PICKS = [("slam", "unseen", 252, "slam"), ("knock", "unseen", 10, "knock")]
ROWS = [("mix", "Unprocessed", "#6b6a66"), ("C", "Original RNN", "#eb6834"), ("H", "Proposed model", "#2a78d6")]
INK, SURF = "#16181d", "#fbfbf8"
LEAD, GAP, TAIL = 0.5, 0.7, 1.0

plt.rcParams.update({"font.family": "DejaVu Sans"})
corpus = dm.load_corpus()
run_dir = tm.run_dir_for(corpus)
rows = select_test_rows(tm.get_recipes(corpus, run_dir), 100)
models = {v: mv.Trained.load(run_dir / v / f"seed{SEED}" / "best.pkl") for v in ("C", "H")}
import pandas as pd
scores = pd.read_csv(run_dir / "multiseed" / f"seed{SEED}_results.csv")
fs = CFG.audio.sample_rate


def render_base(specs, t, f, onset, active, removed, word, poster=False):
    """One still frame: three spectrogram rows, the active one at full contrast. Returns (rgb array, x0, x1, row y-ranges)."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURF)
    vmax = specs["mix"].max()
    axes = []
    lab_h, row_h, gap = 0.075, 0.225, 0.025          # label strip above each spectrogram, then the spectrogram
    for i, (k, lab, col) in enumerate(ROWS):
        y0 = 1 - 0.035 - (i + 1) * (lab_h + row_h) - i * gap
        ax = fig.add_axes([0.02, y0, 0.96, row_h])
        ax.pcolormesh(t, f / 1000, specs[k], vmin=vmax - 70, vmax=vmax, cmap="gray_r", shading="auto", rasterized=True)
        ax.set_ylim(0, 8)
        ax.set_xticks([]); ax.set_yticks([])
        on = poster or i == active
        for s in ax.spines.values():
            s.set_color(col if on else "#d6d5d0"); s.set_linewidth(5 if on else 1.5)
        ax.axvline(onset, color=col if on else "#bdbcb7", lw=2, ls=(0, (4, 4)))
        if not on:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=SURF, alpha=0.62, zorder=5))
        txt = f"{i + 1}  {lab}" + ("" if k == "mix" else f"  ·  {removed[k]:.0f} dB removed")
        fig.text(0.02, y0 + row_h + 0.012, txt, va="bottom", ha="left", fontsize=33,
                 fontweight="bold" if on else "normal", color=INK if on else "#a3a29d")
        if on:
            fig.patches.append(plt.Rectangle((0.005, y0 + row_h + 0.016), 0.008, lab_h - 0.03,
                                             transform=fig.transFigure, color=col, figure=fig))
        axes.append(ax)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    boxes = []
    for ax in axes:
        bb = ax.get_window_extent()
        boxes.append((int(round(H - bb.y1)), int(round(H - bb.y0)), bb.x0, bb.x1))
    plt.close(fig)
    return img, boxes


for name, cond, idx, word in PICKS:
    row = next(r for r in rows if r["cond"] == cond and r["str_db"] == STR_DB and r["speech_idx"] == idx)
    mix = dm.realise(corpus, row)
    n = len(mix["x"])
    fit = lambda y: np.pad(np.asarray(y, np.float32)[:n], (0, max(0, n - len(y))))
    sigs = {"mix": fit(mix["x"]), "C": fit(mv.process_batch("C", [mix], models["C"])[0]["y"]),
            "H": fit(mv.process_batch("H", [mix], models["H"])[0]["y"])}
    g = 0.89 / max(np.abs(v).max() for v in sigs.values())
    sigs = {k: v * g for k, v in sigs.items()}
    sc = scores[(scores.cond == cond) & (scores.str_db == STR_DB) & (scores.speech_idx == idx)].set_index("variant")
    removed = {v: float(sc.loc[v, "ta_db"]) for v in ("C", "H")}
    specs = {}
    for k in sigs:
        f, t, Z = stft(sigs[k], fs=fs, nperseg=512, noverlap=512 - 128)
        specs[k] = 20 * np.log10(np.abs(Z) + 1e-9)
    dur = n / fs
    onset = mix["onset"] / fs
    bases = [render_base(specs, t, f, onset, a, removed, word) for a in range(3)]
    poster, _ = render_base(specs, t, f, onset, 0, removed, word, poster=True)
    plt.imsave(OUT / f"{name}_still.png", poster)
    # audio timeline
    sil = lambda s: np.zeros(int(round(s * fs)), np.float32)
    audio = np.concatenate([sil(LEAD), sigs["mix"], sil(GAP), sigs["C"], sil(GAP), sigs["H"], sil(TAIL)])
    import soundfile as sf
    sf.write(OUT / f"{name}.wav", audio, fs, subtype="PCM_16")
    starts = [LEAD, LEAD + dur + GAP, LEAD + 2 * (dur + GAP)]
    total = len(audio) / fs
    nframes = int(np.ceil(total * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", str(OUT / f"{name}.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
           "-preset", "medium", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
           str(OUT / f"{name}.mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(nframes):
        tt = fi / FPS
        a = 0 if tt < starts[1] - GAP / 2 else (1 if tt < starts[2] - GAP / 2 else 2)
        img, boxes = bases[a]
        frame = img.copy()
        local = tt - starts[a]
        if 0 <= local <= dur:
            y0, y1, x0, x1 = boxes[a]
            x = int(round(x0 + (local / dur) * (x1 - x0)))
            col = np.array([int(ROWS[a][2][i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
            frame[y0:y1, max(0, x - 2):x + 2] = col
        p.stdin.write(frame.tobytes())
    p.stdin.close(); p.wait()
    print(name, cond, idx, row.get("talker", ""), f"dur {dur:.2f}s total {total:.1f}s", removed,
          {v: (round(float(sc.loc[v, 'speech_preserved']), 3), round(float(sc.loc[v, 'stoi']), 3)) for v in ("C", "H")},
          "onset", round(onset, 2), flush=True)
