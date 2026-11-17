# Hearing project: scoring the models - Jake Abraham
"""
PLAIN WORDS
-----------
Measures how well each model did on the frozen test set. Two questions matter for a hearing-aid
wearer, and they pull against each other:

  1. How much quieter did the crinkle get?            -> "transient attenuation" in dB (higher = more comfortable)
  2. How much did the speech get damaged doing it?     -> "speech preserved" (1.0 = untouched)

Any model can win #1 by turning everything down, and win #2 by doing nothing. A good model wins
both at once. The trick that lets us measure them separately: the model produces a mask from the
mixture, and we then apply that *same* mask to the clean speech alone and to the transient alone
("shadow filtering"). Since masking is linear, this splits the output exactly into "what happened to
the speech" and "what happened to the crinkle".

Standard metrics are also reported so results can be compared with the literature: STOI/ESTOI
(intelligibility, what Keshavarzi used) and SI-SDR (quality).

IMPLEMENTATION STATUS
---------------------
IMPLEMENTED: STOI, ESTOI, SI-SDR, transient attenuation, peak-level reduction, speech preserved, speech
  attenuation, parallel scoring, aggregation with 95% confidence intervals.
NOT YET (optional, later): PESQ (the `pesq` package does not build on this Python 3.13 sandbox; it is used
  automatically if installed), HASPI/HASQI via pyclarity for simulated hearing loss using the 10 audiograms
  in Keshavarzi Table 1. (The mask-exponent trade-off sweep lives in run_experiments.tradeoff.)

This module deliberately imports only numpy/pystoi (no JAX) so it can run in worker processes.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from pystoi import stoi as _stoi

FS = 16_000
PRE_PEAK = int(0.050 * FS)          # matches CFG.data.transient_pre_peak_ms
WINDOW_AFTER_PEAK = int(0.200 * FS)

try:
    from pesq import pesq as _pesq   # optional
except ImportError:
    _pesq = None


###########################################################
#                        METRICS                          #
###########################################################

def transient_window(onset: int, n: int) -> slice:
    return slice(onset, min(n, onset + PRE_PEAK + WINDOW_AFTER_PEAK))


def energy_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.sum(np.square(x, dtype=np.float64)) + 1e-20)


def transient_attenuation_db(v: np.ndarray, v_hat: np.ndarray, onset: int) -> float:
    """How much quieter the transient got (dB), measured where it actually is."""
    w = transient_window(onset, len(v))
    return energy_db(v[w]) - energy_db(v_hat[w])


def peak_level_reduction_db(x: np.ndarray, y: np.ndarray, onset: int) -> float:
    """Reduction of the loudest instant of the mixture around the transient (a loudness-spike proxy)."""
    w = transient_window(onset, len(x))
    return 20 * np.log10((np.abs(x[w]).max() + 1e-12) / (np.abs(y[w]).max() + 1e-12))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    return 10 * np.log10(np.sum((a * ref) ** 2) / (np.sum((est - a * ref) ** 2) + 1e-12) + 1e-12)


def score_one(item: dict) -> dict:
    """item: dict(s, v, x, onset, y, s_hat, v_hat, + any metadata). Returns metadata + metrics."""
    s, v, x, y, s_hat, v_hat, onset = (item[k] for k in ("s", "v", "x", "y", "s_hat", "v_hat", "onset"))
    out = {k: item[k] for k in item if k not in ("s", "v", "x", "y", "s_hat", "v_hat", "n")}
    out.update(
        stoi=_stoi(s, y, FS), estoi=_stoi(s, y, FS, extended=True), si_sdr=si_sdr(s, y),
        stoi_in=_stoi(s, x, FS),
        speech_preserved=_stoi(s, s_hat, FS),
        speech_att_db=energy_db(s) - energy_db(s_hat),
        ta_db=transient_attenuation_db(v, v_hat, onset),
        peak_red_db=peak_level_reduction_db(x, y, onset),
    )
    if _pesq is not None:
        try:
            out["pesq"] = _pesq(FS, s, y, "wb")
        except Exception:  # noqa: BLE001 — PESQ fails on some very short/silent inputs
            out["pesq"] = np.nan
    return out


def score_tradeoff(item: dict) -> dict:
    """Only what the trade-off sweep needs: transient attenuation and speech preserved."""
    return dict(ta_db=transient_attenuation_db(item["v"], item["v_hat"], item["onset"]),
                speech_preserved=_stoi(item["s"], item["s_hat"], FS))


def score_many_tradeoff(items: list, workers: int = 4) -> list:
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(score_tradeoff, items, chunksize=8))


def score_many(items: list, workers: int = 4) -> list:
    if workers <= 1:
        return [score_one(i) for i in items]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(score_one, items, chunksize=8))


###########################################################
#                      AGGREGATION                        #
###########################################################

METRICS = ("stoi", "estoi", "si_sdr", "speech_preserved", "speech_att_db", "ta_db", "peak_red_db")


def aggregate(rows: list, by=("variant", "cond", "str_db")):
    """Mean and 95% CI half-width of every metric, grouped by `by`. Returns a pandas DataFrame."""
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = [m for m in METRICS + ("pesq",) if m in df]
    g = df.groupby(list(by))[metrics]
    mean, sem, n = g.mean(), g.sem(), g.size()
    out = mean.add_suffix("_mean").join((1.96 * sem).add_suffix("_ci"))
    out["n"] = n
    return out.reset_index()


