"""Small differentiable checks that run without the LDA backbone or datasets."""

import unittest

import torch
from torch import nn

from lda.model.modules.action_model.wam_ttt import TTTMemoryBank
from lda.training.wam_ttt import WAMTTTMetaLearner, phase_align_human_examples


class TinyFramework(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.action_model = nn.Module()
        self.action_model.model = nn.Module()
        self.action_model.model.ttt_memory = TTTMemoryBank(1, 4, 3, 8)

    def forward(self, examples, *, ttt_fast_weights, return_ttt_inputs=False):
        tokens = self.encoder(torch.stack([item["input"] for item in examples])).unsqueeze(1)
        memory = self.action_model.model.ttt_memory
        prediction = memory.read(0, tokens, ttt_fast_weights[0])
        target = torch.stack([item["target"] for item in examples]).unsqueeze(1)
        loss = (prediction - target).square().mean()
        result = {"loss": loss, "action_loss": loss.detach(), "dynamics_loss": loss.detach()}
        if return_ttt_inputs:
            result["ttt_inputs"] = (tokens,)
        return result


class MetaTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = TinyFramework()
        self.learner = WAMTTTMetaLearner(self.model)
        self.human = [{"input": torch.randn(4), "target": torch.randn(4)} for _ in range(2)]
        self.robot = [{"input": torch.randn(4), "target": torch.randn(4)} for _ in range(2)]

    def test_meta_step_backpropagates_to_slow_and_initial_fast_weights(self):
        output = self.learner(self.human, self.robot)
        output["loss"].backward()
        self.assertIsNotNone(self.model.encoder.weight.grad)
        self.assertTrue(torch.isfinite(self.model.encoder.weight.grad).all())
        initial = self.model.action_model.model.ttt_memory.layers[0].fast_mlp[0].weight
        self.assertIsNotNone(initial.grad)
        self.assertGreater(initial.grad.abs().sum().item(), 0)
        self.assertEqual(len(output["fast_weights"]), 1)

    def test_test_time_adaptation_does_not_modify_model_parameters(self):
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        weights = self.learner.adapt_from_human(self.human)
        self.assertEqual(len(weights), 1)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
            self.assertTrue(parameter.requires_grad)

    def test_phase_alignment_selects_closest_human_frame(self):
        human = [
            {"pair_key": "pour", "episode_phase": 0.1},
            {"pair_key": "pour", "episode_phase": 0.8},
        ]
        robot = [{"pair_key": "pour", "episode_phase": 0.7}]
        paired_human, paired_robot = phase_align_human_examples(human, robot)
        self.assertIs(paired_human[0], human[1])
        self.assertIs(paired_robot[0], robot[0])


if __name__ == "__main__":
    unittest.main()
