# FastWAM: WAM-TTT implementation for LDA

This repository contains the WAM-TTT code we added for the [Latent Dynamics Action (LDA)](https://github.com/jiangranlv/latent-dynamics-action) model: fast-weight memory, human-video inner-loop adaptation, robot-task meta-training, configuration, launch script, and focused tests. The original LDA source and its Git history are not included. `patches/lda-integration.patch` contains the small integration changes needed in LDA.

## Install into LDA

The integration patch targets LDA commit `06e6a274a9086cc26635a9fe663866335eb30fc5`. From a directory containing this repository:

```bash
git clone https://github.com/jiangranlv/latent-dynamics-action.git
git -C latent-dynamics-action checkout 06e6a274a9086cc26635a9fe663866335eb30fc5
git -C latent-dynamics-action apply ../fastwam/patches/lda-integration.patch
cp -a fastwam/lda/. latent-dynamics-action/lda/
cp -a fastwam/scripts/. latent-dynamics-action/scripts/
cp -a fastwam/tests/. latent-dynamics-action/tests/
```

Run the remaining LDA installation steps from its upstream README. The patch changes four upstream files; the other files in this repository are our additions. The upstream LDA code has its own [CC BY-NC 4.0 license](https://github.com/jiangranlv/latent-dynamics-action/blob/main/LICENSE).

## Train

Edit `latent-dynamics-action/lda/config/training/WAM_TTT_LDA.yaml` to set model paths, paired robot and human LeRobot datasets, observation dimensions, and output directory. Use the same normalized task text in corresponding human and robot trajectories. Then, from the LDA directory:

```bash
NUM_PROCESSES=1 bash scripts/run_scripts/run_wam_ttt_LDA.sh
```

Set `NUM_PROCESSES` to the GPU count for distributed training. The default WAM-TTT settings are 48-dimensional memory, 128 hidden units, one inner step, meta learning rate 0.1, test-time learning rate 0.01, and K/V loss weight 0.04. Training checkpoints are written under `run_root_dir/run_id/checkpoints`.

At test time, call `WAMTTTMetaLearner.adapt_from_human(human_examples)` to obtain episode-specific fast weights, then pass them to `model.predict_action(robot_examples, ttt_fast_weights=fast_weights)`.

The paper's paired datasets, pretrained weights, and robot evaluation setup are not included. This is an implementation for compatible data, not a claim that the paper's reported experiments have been reproduced.
