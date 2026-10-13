"""WAM-TTT meta-training and paired LeRobot batch utilities."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from lda.model.modules.action_model.wam_ttt import FastWeightStates, TTTMemoryBank


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def get_ttt_memory(model: nn.Module) -> TTTMemoryBank:
    framework = _unwrap(model)
    memory = framework.action_model.model.ttt_memory
    if memory is None:
        raise ValueError("Enable framework.action_model.diffusion_model_cfg.wam_ttt_enabled")
    return memory


def phase_align_human_examples(
    human_examples: Sequence[dict], robot_examples: Sequence[dict]
) -> tuple[list[dict], list[dict]]:
    """Pair each robot sample with the closest human sample for the same skill.

    ``pair_key`` is the normalized language/skill label and ``episode_phase``
    is the normalized timestep within the episode.  Repeated human samples are
    allowed when robot batches are larger than the available human candidates.
    """
    candidates: dict[str, list[dict]] = {}
    for example in human_examples:
        candidates.setdefault(example["pair_key"], []).append(example)

    paired_human = []
    paired_robot = []
    missing = set()
    for robot in robot_examples:
        skill = robot["pair_key"]
        skill_candidates = candidates.get(skill)
        if not skill_candidates:
            missing.add(skill)
            continue
        human = min(
            skill_candidates,
            key=lambda sample: abs(float(sample["episode_phase"]) - float(robot["episode_phase"])),
        )
        paired_human.append(human)
        paired_robot.append(robot)

    if not paired_robot:
        preview = ", ".join(sorted(missing)[:3]) or "<empty pair_key>"
        raise ValueError(
            "No human/robot skill pairs in this batch. Use matching language labels in both datasets; "
            f"unmatched robot labels include: {preview}"
        )
    return paired_human, paired_robot


class WAMTTTMetaLearner(nn.Module):
    """Compose LDA's human video and robot action objectives into a meta-step."""

    def __init__(
        self,
        model: nn.Module,
        inner_steps: int = 1,
        inner_learning_rate: float = 0.1,
        test_learning_rate: float = 0.01,
        kv_loss_weight: float = 0.04,
    ) -> None:
        super().__init__()
        if inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        if inner_learning_rate <= 0 or test_learning_rate <= 0:
            raise ValueError("inner and test learning rates must be positive")
        if kv_loss_weight < 0:
            raise ValueError("kv_loss_weight cannot be negative")
        self.model = model
        self.inner_steps = inner_steps
        self.inner_learning_rate = inner_learning_rate
        self.test_learning_rate = test_learning_rate
        self.kv_loss_weight = kv_loss_weight

    def forward(self, human_examples: Sequence[dict], robot_examples: Sequence[dict]) -> dict[str, Tensor | FastWeightStates]:
        """Return the robot outer loss and differentiable fast weights."""
        memory = get_ttt_memory(self.model)
        fast_weights = memory.initial_fast_weights()
        inner_video_loss = None
        inner_kv_loss = None

        for _ in range(self.inner_steps):
            human_output = self.model(
                list(human_examples),
                ttt_fast_weights=fast_weights,
                return_ttt_inputs=True,
            )
            inner_video_loss = human_output["loss"]
            inner_kv_loss = memory.kv_loss(human_output["ttt_inputs"], fast_weights)
            inner_loss = inner_video_loss + self.kv_loss_weight * inner_kv_loss
            fast_weights = memory.sgd_step(
                inner_loss,
                fast_weights,
                learning_rate=self.inner_learning_rate,
                create_graph=True,
            )

        robot_output = self.model(list(robot_examples), ttt_fast_weights=fast_weights)
        return {
            "loss": robot_output["loss"],
            "action_loss": robot_output["action_loss"],
            "dynamics_loss": robot_output["dynamics_loss"],
            "inner_video_loss": inner_video_loss.detach(),
            "inner_kv_loss": inner_kv_loss.detach(),
            "fast_weights": fast_weights,
        }

    @torch.enable_grad()
    def adapt_from_human(self, human_examples: Sequence[dict]) -> FastWeightStates:
        """Adapt only fast weights from human video, for a single test episode."""
        memory = get_ttt_memory(self.model)
        fast_weights = memory.initial_fast_weights(detach=True)
        modules = tuple(self.model.modules())
        original_training = tuple(module.training for module in modules)
        parameters = tuple(self.model.parameters())
        original_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        self.model.eval()
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            for _ in range(self.inner_steps):
                human_output = self.model(
                    list(human_examples),
                    ttt_fast_weights=fast_weights,
                    return_ttt_inputs=True,
                )
                kv_loss = memory.kv_loss(human_output["ttt_inputs"], fast_weights)
                inner_loss = human_output["loss"] + self.kv_loss_weight * kv_loss
                fast_weights = memory.sgd_step(
                    inner_loss,
                    fast_weights,
                    learning_rate=self.test_learning_rate,
                    create_graph=False,
                )
        finally:
            for parameter, requires_grad in zip(parameters, original_requires_grad, strict=True):
                parameter.requires_grad_(requires_grad)
            for module, training in zip(modules, original_training, strict=True):
                module.train(training)
        return fast_weights
