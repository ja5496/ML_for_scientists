# Hearing project: real-time (streaming) processing - Jake Abraham
"""
PLAIN WORDS
-----------
Runs a trained model the way a hearing aid would: audio arrives 2.5 ms at a time, and each chunk
is processed and sent out immediately, using only what has already been heard. This file checks two
things a hearing aid must satisfy:
  1. The streaming output is identical to the offline (whole-file) output — proof that nothing
     secretly used the future during training or evaluation.
  2. The delay stays at ~7.5 ms (Keshavarzi's figure). Compute per chunk is MEASURED and reported, not
     constrained, during the proof of concept (the full-size belief network is far beyond a hearing-aid chip).

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
StreamState: input sample buffer (last frame_len samples), output overlap-add buffer,
             per-LSN ring buffers + pooling accumulators + attention EMA stats, LSTM carries

process_hop(state, new_samples (hop_len,)) -> state, out_samples (hop_len,):
    1. append new samples; form the newest 5-ms frame; window; rfft
    2. band energies -> log features (with stored training stats)
    3. advance DBN root by one frame; any LSN whose pooling block completes emits a frame and
       advances its children (recursively); run attention for each LSN that emitted
    4. gather held outputs of all 15 LSNs -> mask_estimator_lstm.step(...) -> mask (64,)
    5. apply mask to the current spectrum, irfft, overlap-add, emit the oldest hop_len samples

run_file(model, params, x) -> y_stream
check_equivalence(model, params, test_files): max |y_stream - y_offline| < 1e-4 for every file
measure():
    algorithmic latency (samples from input sample to its influence on output) -> expect 120 samples (7.5 ms)
    MACs per hop for each variant (analytic count from layer shapes)
    wall-clock per hop on CPU (single thread) -> real-time factor
    write runs/cost_table.csv : variant, params, MACs/hop, latency_ms, RTF
"""


def init_state(model, params):
    raise NotImplementedError


def process_hop(model, params, state, new_samples):
    raise NotImplementedError


def run_file(model, params, x):
    raise NotImplementedError


def check_equivalence(model, params, files, tol: float = 1e-4):
    raise NotImplementedError


def measure(model, params):
    raise NotImplementedError
