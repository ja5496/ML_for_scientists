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
(intelligibility, what Keshavarzi used), PESQ and SI-SDR (quality), and HASPI/HASQI, which simulate
a hearing-impaired listener (closest thing to grandma we can test without a listening study).

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
Libraries: pystoi (STOI/ESTOI), pesq (wideband), pyclarity (HASPI v2, HASQI v2, audiograms),
           numpy for SI-SDR. Pin versions in requirements.txt.

shadow_filter(M, s, v, n):
    s_hat = istft(apply_mask(stft(s), M));  v_hat = ...(v);  n_hat = ...(n);  y = s_hat + v_hat + n_hat

transient_attenuation_db(v, v_hat, onset):
    window = [onset - 5 ms, onset + 200 ms]
    return 10 log10( sum v[window]^2 / sum v_hat[window]^2 )

peak_level_reduction_db(x, y, onset): 20 log10(max|x[window]| / max|y[window]|)   # loudness-spike proxy

speech_preserved(s, s_hat): STOI(s_ref=s, s_hat)        # 1.0 = speech untouched by the mask
speech_attenuation_db(s, s_hat): 10 log10(sum s^2 / sum s_hat^2)   (whole utterance)

score_one(model, mixture) -> dict:
    y, M = model.enhance(mixture.x)
    s_hat, v_hat, n_hat = shadow_filter(M, ...)
    return dict(stoi=STOI(s, y), estoi=ESTOI(s, y), pesq=PESQ(s, y), si_sdr=SI_SDR(s, y),
                ta_db=transient_attenuation_db(...), peak_red_db=..., speech_preserved=...,
                speech_att_db=..., haspi=HASPI(s, y, audiogram), hasqi=HASQI(...))
    audiograms: use the 10 HI audiograms in Keshavarzi Table 1 (public in the paper) + normal hearing

tradeoff_curve(model, test_set, exponents=(0.25, 0.5, 0.75, 1, 1.5, 2, 3)):
    for p in exponents: M_p = M ** p (p>1 = more aggressive) -> (mean ta_db, mean speech_preserved)
    This traces each model's attenuation-vs-damage frontier — the left panel of the money plot.

aggregate(results) -> DataFrame grouped by variant x condition (seen/unseen/grandma) x STR x background,
                      mean ± 95% CI over utterances; paired Wilcoxon H vs each of C, D, E (Holm-corrected).
"""


def shadow_filter(M, s, v, n):
    raise NotImplementedError


def transient_attenuation_db(v, v_hat, onset, fs: int = 16_000):
    raise NotImplementedError


def speech_preserved(s, s_hat, fs: int = 16_000):
    raise NotImplementedError


def score_one(model, params, mixture: dict, audiogram=None) -> dict:
    raise NotImplementedError


def tradeoff_curve(model, params, test_set, exponents=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    raise NotImplementedError


def aggregate(results: list):
    raise NotImplementedError
