# Hearing project: tests that nothing peeks into the future - Jake Abraham
"""
PLAIN WORDS
-----------
A hearing aid cannot know what sound is coming next. If any part of the model accidentally uses
future audio (very easy to do with slow, pooled layers), it will look better in testing than it
could ever be in a real device. These tests change the *future* part of an input and check that the
*past* outputs don't move. Run them after implementing each module — before trusting any result.

PSEUDOCODE FOR THE IMPLEMENTING AGENT (pytest)
----------------------------------------------
helper assert_causal(fn, x, t0, out_frame_of_sample):
    y1 = fn(x); x2 = x.at[t0:].set(random noise); y2 = fn(x2)
    assert allclose(y1[:last output frame that depends only on samples < t0], y2[same], atol=1e-6)

test_frontend_causal          : stft frames
test_crbm_causal              : dbn_crbm.hidden_preact, prob_max_pool for pool in (1, 3)
test_hold_to_base_causal      : dbn_distributed.hold_to_base for rates 1, 3, 9, 27
                                + exact check: slow frame k first appears at base frame (k+1)*R - 1
test_dbn_tree_causal          : dbn_distributed.forward, every LSN
test_attention_causal         : attention_feedback.apply
test_mask_net_causal          : mask_estimator_lstm.apply mode="stream" and "window3"
test_full_model_causal        : model_variants.build("H").enhance end-to-end, on samples
test_stream_equals_offline    : enhance_stream.check_equivalence
"""

import pytest


@pytest.mark.skip(reason="implement after modules exist")
def test_full_model_causal():
    raise NotImplementedError
