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
  6. (Later, only if H wins) compression: full size (300 units) vs HEARING_AID_SIZE (48 units) / distilled student
  7. CRBM vs sparse autoencoder DBN backend
  8. Frozen vs fine-tuned belief network

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
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


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
