#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-lda/config/training/WAM_TTT_LDA.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-8}

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  lda/training/train_wam_ttt.py \
  --config_yaml "${CONFIG}" \
  "$@"
