# Hearing project: the "money plot" - Jake Abraham
"""
PLAIN WORDS
-----------
One figure, two panels, that should convince a skeptic in ten seconds.

LEFT — "More crinkle removed, less speech damaged."
    x: how much quieter the transient got (dB). y: how intact the speech is (1.0 = untouched).
    Each model traces a curve as we make it more or less aggressive. Top-right is perfect.
    A model is strictly better if its whole curve sits up-and-right of another's. The arrow reads off
    the headline number: at the same speech quality, how many extra dB of crinkle the proposed model removes.
    Run on transient types the models never heard during training, at −15 dB STR.

RIGHT — "The advantage grows exactly where hearing aids struggle."
    x: how loud the transient is relative to speech (STR; further right = harsher).
    y: predicted intelligibility (STOI) of the processed output.
    Shaded region = loudness range seen in training; everything right of it is new to every model.
    Best case: the proposed model degrades gracefully and stays near the perfect-mask ceiling while
    the original RNN falls away — mirroring Bellur & Elhilali's finding that distributed beliefs help
    most in mismatched conditions.

The size-and-context-matched RNN (variant E) is in both panels on purpose: beating it rules out
"it only won because it was bigger / had more memory".

Right now (mode="sketch") the numbers are HYPOTHETICAL best-case values, chosen to show what the
plot would look like if the hypothesis is right. The figure is stamped accordingly. After the
experiments run, call make_money_plot(from_results=True) to draw the same figure from real data.

PSEUDOCODE FOR THE IMPLEMENTING AGENT (from_results=True)
---------------------------------------------------------
left  <- runs/results/tradeoff_unseen_str-15.csv : variant, exponent, ta_db_mean, ta_db_ci, sp_mean, sp_ci
right <- runs/results/stoi_vs_str_unseen.csv     : variant, str_db, stoi_mean, stoi_ci
Same styling; draw 95% CI as error bars (right) and as small crosses at the p=1 operating points (left).
Compute the arrow from data: y_ref = speech_preserved of variant C at p=1; interpolate each frontier at y_ref.
Remove the "hypothetical" stamp only in this mode.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots (dataviz reference palette, light mode) + neutrals.
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
STYLE = {
    "H": dict(label="Proposed: filterbank + belief network + attention", short="Proposed (H)",
              color="#2a78d6", lw=3.0, marker="o", ls="-", z=6),
    "E": dict(label="LSTM, size- & context-matched to proposed", short="LSTM matched (E)",
              color="#1baf7a", lw=2.0, marker="s", ls="-", z=5),
    "C": dict(label="Keshavarzi et al. 2021 RNN (original)", short="Original RNN (C)",
              color="#eb6834", lw=2.0, marker="^", ls="-", z=4),
    "B": dict(label="MCTR (non-neural)", short="MCTR (B)",
              color=MUTED, lw=1.6, marker="D", ls="-", z=3),
    "A": dict(label="No processing", short="No processing",
              color=MUTED, lw=1.6, marker="", ls=":", z=2),
    "I": dict(label="Ideal ratio mask (ceiling)", short="Ideal mask (ceiling)",
              color=INK, lw=1.6, marker="*", ls="--", z=3),
}
# Ablations of H: neutral greys told apart by line style + direct labels (only 3 hues stay
# colour-blind-safe when lines cross, and those are reserved for H, E and C).
ABLATION_STYLE = {
    "G": dict(label="Proposed without attention (G)", short="No attention (G)",
              color=INK_2, lw=1.8, marker="v", ls="-.", z=4),
    "F": dict(label="Proposed without the filterbank (F)", short="No filterbank (F)",
              color=MUTED, lw=1.8, marker="P", ls=":", z=4),
}


###########################################################
#          HYPOTHETICAL BEST-CASE NUMBERS (SKETCH)        #
###########################################################

def sketch_tradeoff():
    """Speech preserved = 1 - a * (attenuation / x0) ** q : each curve is one model's frontier."""
    shapes = {"B": (0.20, 12.0, 1.5, 12.0), "C": (0.18, 20.0, 2.0, 24.0),
              "E": (0.18, 23.0, 2.2, 26.0), "H": (0.12, 28.0, 3.0, 30.0)}
    op_points = {"B": 6.0, "C": 12.0, "E": 14.0, "H": 20.0}           # where the unmodified mask (p = 1) lands
    curves = {}
    for v, (a, x0, q, xmax) in shapes.items():
        x = np.linspace(0, xmax, 200)
        f = lambda t, a=a, x0=x0, q=q: 1 - a * (t / x0) ** q
        curves[v] = dict(x=x, y=f(x), op=(op_points[v], f(op_points[v])), f=f, a=a, x0=x0, q=q)
    return curves, (27.0, 0.99)                                        # IRM point


def sketch_stoi_vs_str():
    strs = np.array([-5, -10, -15, -20, -25, -30])
    return strs, {
        "A": [0.84, 0.76, 0.67, 0.59, 0.52, 0.46],
        "B": [0.85, 0.78, 0.70, 0.62, 0.55, 0.49],
        "C": [0.89, 0.84, 0.78, 0.71, 0.63, 0.56],
        "E": [0.90, 0.86, 0.80, 0.74, 0.66, 0.59],
        "H": [0.92, 0.90, 0.87, 0.84, 0.80, 0.76],
        "I": [0.97, 0.96, 0.95, 0.94, 0.93, 0.92],
    }


###########################################################
#                          PLOT                           #
###########################################################

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def make_money_plot(from_results: bool = False, out_path: Path = Path(__file__).parent / "figures" / "money_plot_sketch.png",
                    run_dir: Path = None):
    if from_results:
        return make_money_plot_from_results(run_dir)

    curves, irm_pt = sketch_tradeoff()
    strs, stoi = sketch_stoi_vs_str()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=SURFACE,
                                   gridspec_kw=dict(wspace=0.30))

    # ---------------- LEFT: trade-off frontier ----------------
    _style_axes(axL)
    for v in ("B", "C", "E", "H"):
        s, c = STYLE[v], curves[v]
        axL.plot(c["x"], c["y"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=s["z"], label=s["label"],
                 solid_capstyle="round")
        axL.plot(*c["op"], marker=s["marker"], ms=9, color=s["color"], mec=SURFACE, mew=2, zorder=s["z"] + 1)
    # direct labels placed along each curve (solve curve for a chosen y), white backing for legibility
    for v, y_lab, side in (("B", 0.83, "right"), ("C", 0.88, "left"), ("E", 0.835, "right"), ("H", 0.86, "right")):
        c = curves[v]
        x_lab = c["x0"] * ((1 - y_lab) / c["a"]) ** (1 / c["q"])
        axL.annotate(STYLE[v]["short"], (x_lab, y_lab), xytext=(8 if side == "right" else -8, 0),
                     textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                     fontsize=10, color=INK, zorder=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    axL.plot(0, 1.0, marker="o", ms=8, color=MUTED, mec=SURFACE, mew=2, zorder=7)
    axL.annotate("No processing\n(start point)", xy=(0, 1.0), xytext=(0.4, 0.905), fontsize=10, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=5))
    axL.plot(*irm_pt, marker="*", ms=16, color=INK, mec=SURFACE, mew=1.5, zorder=7, label=STYLE["I"]["label"])
    axL.annotate("Ideal mask", irm_pt, xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=10, color=INK)

    # headline arrow at equal speech quality
    y_ref = 0.95
    xs = {v: curves[v]["x0"] * ((1 - y_ref) / curves[v]["a"]) ** (1 / curves[v]["q"]) for v in ("C", "H")}
    axL.axhline(y_ref, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    axL.annotate("", xy=(xs["H"], y_ref), xytext=(xs["C"], y_ref),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=16), zorder=9)
    axL.annotate(f"+{xs['H'] - xs['C']:.0f} dB more transient removed\nat the same speech quality",
                 xy=((xs["C"] + xs["H"]) / 2, y_ref), xytext=(30.5, 0.968), ha="center", va="bottom",
                 fontsize=10.5, color=INK, fontweight="bold", zorder=9,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=4,
                                 connectionstyle="arc3,rad=0.15"))

    axL.set_xlim(-1, 38)
    axL.set_ylim(0.80, 1.005)
    axL.set_xlabel("Transient attenuation (dB)   →  more comfortable", fontsize=11, color=INK)
    axL.set_ylabel("Speech preserved (STOI of masked clean speech)", fontsize=11, color=INK)
    axL.set_title("A   More crinkle removed, less speech damaged", loc="left", fontsize=12.5,
                  color=INK, fontweight="bold", pad=24)
    axL.text(0, 1.02, "Unseen transient types, STR −15 dB. Dots = default setting; curves = more/less aggressive.",
             transform=axL.transAxes, fontsize=9.5, color=INK_2)
    axL.text(37.5, 0.803, "better ↗", ha="right", va="bottom", fontsize=10, color=INK_2, style="italic")

    # ---------------- RIGHT: STOI vs STR ----------------
    _style_axes(axR)
    axR.axvspan(-15.5, -4.5, color=GRID, alpha=0.6, zorder=0, lw=0)
    axR.text(-10, 0.425, "trained on\nthis range", ha="center", va="bottom", fontsize=9.5, color=INK_2)
    axR.text(-22.5, 0.425, "never seen in training  →", ha="center", va="bottom", fontsize=9.5, color=INK_2)
    for v in ("A", "B", "C", "E", "I", "H"):
        s = STYLE[v]
        axR.plot(strs, stoi[v], color=s["color"], lw=s["lw"], ls=s["ls"], marker=s["marker"] or None,
                 ms=7 if v != "I" else 10, mec=SURFACE, mew=1.5, zorder=s["z"], label=s["label"])
    # direct labels on the right end
    label_y = {"I": 0.92, "H": 0.76, "E": 0.605, "C": 0.555, "B": 0.495, "A": 0.455}
    for v, y in label_y.items():
        axR.annotate(STYLE[v]["short"], (-30, y), xytext=(10, 0), textcoords="offset points",
                     va="center", fontsize=10, color=INK)
    # gap annotation at -30 dB
    axR.annotate("", xy=(-29.2, stoi["H"][-1] - 0.005), xytext=(-29.2, stoi["C"][-1] + 0.005),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    axR.text(-28.9, 0.705, f"+{stoi['H'][-1] - stoi['C'][-1]:.2f} STOI",
             fontsize=10.5, color=INK, fontweight="bold", va="center", ha="right")
    axR.set_xlim(-4, -37.5)          # harsher to the right
    axR.set_xticks(strs)
    axR.set_ylim(0.42, 1.0)
    axR.set_xlabel("Speech-to-transient ratio (dB)   →  louder transient", fontsize=11, color=INK)
    axR.set_ylabel("Predicted intelligibility (STOI)", fontsize=11, color=INK)
    axR.set_title("B   The advantage grows where hearing aids struggle", loc="left", fontsize=12.5,
                  color=INK, fontweight="bold", pad=24)
    axR.text(0, 1.02, "Unseen transient types and unseen talkers. Higher is better.",
             transform=axR.transAxes, fontsize=9.5, color=INK_2)

    handles, labels = axR.get_legend_handles_labels()
    order = [labels.index(STYLE[v]["label"]) for v in ("H", "E", "C", "B", "A", "I")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06), labelcolor=INK)

    fig.suptitle("SKETCH — hypothetical best-case values, not experimental results",
                 x=0.5, y=1.03, fontsize=12.5, color="#b8322f", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


###########################################################
#                 THE SAME FIGURE, FROM REAL DATA         #
###########################################################

def _x_at(curve, y_ref):
    """Transient attenuation where a frontier crosses speech-preserved = y_ref (linear interpolation)."""
    c = curve.sort_values("p")
    xs, ys = c.ta_db_mean.to_numpy(), c.sp_mean.to_numpy()
    for i in range(len(xs) - 1):
        if (ys[i] - y_ref) * (ys[i + 1] - y_ref) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (y_ref - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return np.nan


def _seed_tables(run_dir: Path):
    """Every training seed on disk. Seed 0 = the stage-4 files (all variants); seeds 1+ = runs/stage1/multiseed/
    (written by `run_experiments.py multiseed`). Returns (trade-off rows, per-mixture results), both with a seed column."""
    import pandas as pd
    run_dir = Path(run_dir)
    to = [pd.read_csv(run_dir / "tradeoff_unseen_str-15.csv").assign(seed=0)]
    res = [pd.read_csv(run_dir / "stage4_results_per_utterance.csv").assign(seed=0)]
    for f in sorted((run_dir / "multiseed").glob("seed*_tradeoff.csv")):
        k = int(f.name[4:].split("_")[0])
        r = f.with_name(f"seed{k}_results.csv")
        if k == 0 or not r.exists():
            continue
        to.append(pd.read_csv(f).assign(seed=k))
        res.append(pd.read_csv(r).assign(seed=k))
    return pd.concat(to, ignore_index=True), pd.concat(res, ignore_index=True)


def _mean_curves(to):
    """Average each model's trade-off curve over its seeds, point by point (same exponent p)."""
    return (to.groupby(["variant", "p"], as_index=False)
              .agg(ta_db_mean=("ta_db_mean", "mean"), sp_mean=("sp_mean", "mean"), n_seeds=("seed", "nunique")))


def _gain_over_seeds(to, a, b, y_ref):
    """Per-seed extra attenuation of model a over model b where both cross speech-preserved = y_ref."""
    g = []
    for s in sorted(set(to[to.variant == a].seed) & set(to[to.variant == b].seed)):
        g.append(_x_at(to[(to.variant == a) & (to.seed == s)], y_ref) - _x_at(to[(to.variant == b) & (to.seed == s)], y_ref))
    return np.array([x for x in g if np.isfinite(x)])


def make_money_plot_from_results(run_dir: Path = None, out_path: Path = Path(__file__).parent / "figures" / "money_plot_results.png"):
    """Left: attenuation-vs-speech frontier (unseen transients, STR -15 dB). Right: STOI vs STR (unseen).
    C, E and H are averaged over every training seed on disk (seed 0 = stage 4; seeds 1+ = runs/stage1/multiseed/)."""
    import pandas as pd
    from config import CFG
    run_dir = Path(run_dir or CFG.out_dir / "stage1")
    smoke = run_dir.name == "smoke"
    if smoke:                                   # never let a smoke-test figure land where real results go
        out_path = run_dir / "money_plot_SMOKE.png"
    sm = pd.read_csv(run_dir / "stage4_summary.csv")
    cell = lambda v, cond="unseen", st=-15.0: sm[(sm.variant == v) & (sm.cond == cond) & (sm.str_db == st)]
    if smoke:
        to_all = pd.read_csv(run_dir / "tradeoff_unseen_str-15.csv").assign(seed=0)
        res_all = None
    else:
        to_all, res_all = _seed_tables(run_dir)
    to = _mean_curves(to_all)                       # one curve per model = mean over its training seeds
    n_seeds = to_all.groupby("variant").seed.nunique().to_dict()
    n_max = max(n_seeds.get(v, 1) for v in ("C", "E", "H"))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=SURFACE, gridspec_kw=dict(wspace=0.30))
    _style_axes(axL)
    for v in ("C", "E", "H"):
        c = to[to.variant == v].sort_values("p")
        if c.empty:
            continue
        st = STYLE[v]
        if n_seeds.get(v, 1) > 1:                   # thin = each training seed; thick = their mean
            for _, cs in to_all[to_all.variant == v].groupby("seed"):
                cs = cs.sort_values("p")
                axL.plot(cs.ta_db_mean, cs.sp_mean, color=st["color"], lw=0.9, alpha=0.35, zorder=st["z"] - 1)
        axL.plot(c.ta_db_mean, c.sp_mean, color=st["color"], lw=st["lw"], zorder=st["z"], label=st["label"], marker=".", ms=5)
        op = c[c.p == 1.0]
        axL.plot(op.ta_db_mean, op.sp_mean, marker=st["marker"], ms=10, color=st["color"], mec=SURFACE, mew=2, zorder=st["z"] + 1)
        if v == "C":   # label mid-curve, on its open left side (E/H pass right of C's end, the text box sits below)
            k = c[c.p == 2.0] if (c.p == 2.0).any() else c.iloc[[-1]]
            axL.annotate(st["short"], (float(k.ta_db_mean.iloc[0]), float(k.sp_mean.iloc[0])), xytext=(-10, 0),
                         textcoords="offset points", va="center", ha="right", fontsize=10, color=INK)
        else:
            axL.annotate(st["short"], (c.ta_db_mean.iloc[-1], c.sp_mean.iloc[-1]), xytext=(6, 0),
                         textcoords="offset points", va="center", ha="left", fontsize=10, color=INK)
    for v, mk, lbl, off in (("B", "D", "MCTR (B)", (8, -16)), ("I", "*", "Ideal mask", (7, -14))):
        d = cell(v)
        if not d.empty:
            axL.plot(d.ta_db_mean, d.speech_preserved_mean, marker=mk, ms=12 if v == "I" else 8, color=STYLE[v]["color"],
                     mec=SURFACE, mew=1.5, zorder=7, ls="none", label=STYLE[v]["label"] if v == "I" else None)
            axL.annotate(lbl, (float(d.ta_db_mean.iloc[0]), float(d.speech_preserved_mean.iloc[0])), xytext=off,
                         textcoords="offset points", fontsize=10, color=INK)
    axL.plot(0, 1.0, marker="o", ms=8, color=MUTED, mec=SURFACE, mew=2, zorder=7)
    axL.annotate("No processing", (0, 1.0), xytext=(8, 4), textcoords="offset points", fontsize=10, color=INK_2)
    c_op = to[(to.variant == "C") & (to.p == 1.0)]
    if not c_op.empty and "H" in set(to.variant):
        y_ref = float(c_op.sp_mean.iloc[0])
        xc, xh = _x_at(to[to.variant == "C"], y_ref), _x_at(to[to.variant == "H"], y_ref)
        xe = _x_at(to[to.variant == "E"], y_ref) if "E" in set(to.variant) else np.nan
        if np.isfinite(xc) and np.isfinite(xh):
            axL.axhline(y_ref, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
            axL.annotate("", xy=(xh, y_ref), xytext=(xc, y_ref),
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=16), zorder=9)
            gc, ge = _gain_over_seeds(to_all, "H", "C", y_ref), _gain_over_seeds(to_all, "H", "E", y_ref)
            rng = lambda g: f"   (per seed: {g.min():+.1f} to {g.max():+.1f})" if len(g) > 1 else ""
            msg = ("At equal speech quality (dotted line), H removes"
                   + (f" (mean of {len(gc)} seeds)" if len(gc) > 1 else "") + "\n"
                   f"{gc.mean():+.1f} dB more transient than the original RNN (C){rng(gc)}")
            if np.isfinite(xe) and len(ge):
                msg += f"\n{ge.mean():+.1f} dB more than the size-matched LSTM (E){rng(ge)}"
            y0, y1 = axL.get_ylim()                 # free band under the curves so the box never covers them
            lo = float(to_all[to_all.variant.isin(["C", "E", "H"])].sp_mean.min())
            axL.set_ylim(min(y0, lo - 0.42 * (y1 - lo)), y1)
            axL.text(0.03, 0.03, msg, transform=axL.transAxes, ha="left", va="bottom", fontsize=10, color=INK,
                     linespacing=1.5, zorder=9,
                     bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID, lw=1))
    axL.set_xlabel("Transient attenuation (dB)   →  more comfortable", fontsize=11, color=INK)
    axL.set_ylabel("Speech preserved (STOI of masked clean speech)", fontsize=11, color=INK)
    axL.set_title("A   Transient removed vs speech kept", loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=40)
    sub = "Unseen transient types, STR −15 dB. Big marker = model as trained;\nline = same model made gentler / harsher."
    if n_max > 1:
        sub += f" Thick = mean of {n_max} training seeds, thin = each seed."
    axL.text(0, 1.02, sub, transform=axL.transAxes, fontsize=9.5, color=INK_2, va="bottom")

    _style_axes(axR)
    strs = sorted(set(sm.str_db), reverse=True)
    lo, hi = min(CFG.data.train_str_db), max(CFG.data.train_str_db)
    axR.axvspan(hi + 0.5, lo - 0.5, color=GRID, alpha=0.6, zorder=0, lw=0)
    ends = []
    for v in ("A", "B", "C", "E", "I", "H"):
        d = sm[(sm.variant == v) & (sm.cond == "unseen")].sort_values("str_db", ascending=False)
        if res_all is not None and n_seeds.get(v, 1) > 1:   # average each mixture over seeds, then CI over mixtures
            r = res_all[(res_all.variant == v) & (res_all.cond == "unseen")]
            per_mix = r.groupby(["str_db", "speech_idx"]).stoi.mean().groupby("str_db")
            d = pd.DataFrame(dict(stoi_mean=per_mix.mean(), stoi_ci=1.96 * per_mix.std(ddof=1) / np.sqrt(per_mix.size())))
            d = d.reset_index().sort_values("str_db", ascending=False)
        if d.empty:
            continue
        st = STYLE[v]
        axR.errorbar(d.str_db, d.stoi_mean, yerr=d.stoi_ci, color=st["color"], lw=st["lw"], ls=st["ls"],
                     marker=st["marker"] or None, ms=7, mec=SURFACE, capsize=2, zorder=st["z"], label=st["label"])
        ends.append([float(d.stoi_mean.iloc[-1]), st["short"], float(d.str_db.iloc[-1])])
    ylo, yhi = axR.get_ylim()
    gap = 0.045 * (yhi - ylo)                                   # push end labels apart so they never overlap
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, lab, x in ends:
        axR.annotate(lab, (x, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    axR.set_xlim(max(strs) + 1, min(strs) - 9)
    axR.set_xticks(strs)
    axR.set_xlabel("Speech-to-transient ratio (dB)   →  louder transient", fontsize=11, color=INK)
    axR.set_ylabel("Predicted intelligibility (STOI)", fontsize=11, color=INK)
    axR.set_title("B   Intelligibility as the transient gets louder", loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=40)
    axR.text(0, 1.02, "Unseen transient types and talkers. Shaded = loudness range used\nin training. Mean ± 95% CI"
             + (" (C, E, H: each sentence averaged over seeds)." if n_max > 1 else "."),
             transform=axR.transAxes, fontsize=9.5, color=INK_2, va="bottom")
    h, l = axR.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06), labelcolor=INK)
    seeds_txt = f"{n_max} training seeds" if n_max > 1 else "1 training seed"
    fig.suptitle(f"Experimental results — first pass ({seeds_txt}, 100 test sentences per cell, CPU budget)",
                 x=0.5, y=1.06, fontsize=12.5, color=INK, fontweight="bold")
    if smoke:
        fig.text(0.5, 0.5, "SYNTHETIC SMOKE TEST — NOT RESULTS", ha="center", va="center", rotation=20,
                 fontsize=34, color="#b8322f", alpha=0.25, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    if not smoke:
        fig.savefig(run_dir / out_path.name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def make_frontier_plot(run_dir: Path, out_path: Path, variants=("C", "E", "F", "G", "H")) -> Path:
    """Ablation frontier: transient removed vs speech kept for H and its ablations (unseen, STR -15 dB).
    Raw averages can mislead (a gentler model removes less AND keeps more); the frontier compares at equal quality."""
    to_all, _ = _seed_tables(run_dir)
    to = _mean_curves(to_all)                       # mean over each model's training seeds
    n_seeds = to_all.groupby("variant").seed.nunique().to_dict()
    style = {**STYLE, **ABLATION_STYLE}
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor=SURFACE)
    _style_axes(ax)
    plotted = []
    for v in variants:
        c = to[to.variant == v].sort_values("p")
        if c.empty:
            continue
        plotted.append(v)
        st = style[v]
        lab = st["short"] + (f" · {n_seeds[v]} seeds" if n_seeds[v] > 1 else " · 1 seed")
        ax.plot(c.ta_db_mean, c.sp_mean, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"],
                label=lab, marker=".", ms=5)
        op = c[c.p == 1.0]
        ax.plot(op.ta_db_mean, op.sp_mean, marker=st["marker"], ms=10, color=st["color"], mec=SURFACE, mew=2,
                zorder=st["z"] + 1, ls="none")
    ax.set_xlabel("Transient attenuation (dB)   →  more comfortable", fontsize=11, color=INK)
    ax.set_ylabel("Speech preserved (STOI of masked clean speech)", fontsize=11, color=INK)
    ax.set_title("Ablations at equal speech quality", loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=50)
    ax.text(0, 1.02, "Unseen transient types, STR −15 dB. Big marker = model as trained; line = mask ** p, p = 0.5 … 3,\n"
            "averaged over training seeds. Further right at the same height = more transient removed for the same speech damage.\n"
            "No attention / No filterbank = the proposed model (H) with that one part removed.",
            transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")
    h, l = ax.get_legend_handles_labels()           # 5 crossing lines: legend (best first), not end labels
    x97 = [_x_at(to[to.variant == v], 0.97) for v in plotted]
    order = sorted(range(len(h)), key=lambda i: -x97[i] if np.isfinite(x97[i]) else 0)
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print("saved", make_money_plot())
