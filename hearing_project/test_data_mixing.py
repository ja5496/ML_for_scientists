# Hearing project: tests for mixing and recipes - Jake Abraham
"""Run with:  cd hearing_project && python -m pytest -q test_data_mixing.py"""

import numpy as np
import pytest

import data_mixing as dm
import synthetic_audio

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def corpus():
    return dm.load_corpus(synthetic_audio.build(seed=3, talkers=(3, 2, 2), utts=(3, 2, 2), per_class=3))


def test_str_is_exact():
    s = rng.standard_normal(32_000).astype(np.float32)
    t = np.zeros(8_000, np.float32); t[800:1200] = np.exp(-np.arange(400) / 50) * rng.standard_normal(400)
    for target in (-5, -15, -30):
        m = dm.make_mixture(s, t, target, onset=5_000)
        assert abs(dm.measured_str_db(m["s"], m["v"]) - target) < 0.1
        assert np.allclose(m["x"], m["s"] + m["v"] + m["n"])


def test_batches_contain_the_transient(corpus):
    b = next(dm.batch_iterator(corpus, "train", batch_size=4, crop_frames=400, seed=1))
    assert b["x"].shape == (4, 400 * 40)
    for i in range(4):
        assert np.abs(b["v"][i]).max() > 0          # the transient's peak is inside every crop


def test_recipes_are_reproducible(corpus):
    r1, r2 = dm.build_recipes(corpus, seed=7), dm.build_recipes(corpus, seed=7)
    assert r1 == r2 and r1["synthetic"]
    row = r1["rows"][-1]
    a, b = dm.realise(corpus, row), dm.realise(corpus, row)
    assert np.array_equal(a["x"], b["x"])
    assert {r["cond"] for r in r1["rows"]} == {"val", "seen", "unseen", "grandma"}
