#!/usr/bin/env bash
# Hearing project: run stages 2–4 in order (CPU is fine). Re-running resumes: every pretrained
# sub-network and every trained model is checkpointed and skipped if it already exists.
#   stage 2  dbn_pretrain.py            ~2–2.5 h   unsupervised belief network (15 sub-networks)
#   stage 3  run_experiments stage3     ~10 min    speech / transient memories + go/no-go
#   stage 4  run_experiments stage4     ~2.5–3 h   train E F G H, score all nine models, money plot
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs/logs
python3 -u dbn_pretrain.py           2>&1 | tee -a runs/logs/dbn_pretrain.log
python3 -u run_experiments.py stage3 2>&1 | tee runs/logs/stage3.log
python3 -u run_experiments.py stage4 2>&1 | tee runs/logs/stage4.log
echo PIPELINE_DONE
