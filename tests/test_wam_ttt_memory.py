from __future__ import annotations

import unittest

import torch

from lda.model.modules.action_model.wam_ttt import TTTMemoryBank, TTTMemoryLayer


class TTTMemoryLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.layer = TTTMemoryLayer(model_dim=6, memory_dim=3, hidden_dim=8)
        self.tokens = torch.randn(2, 5, 6)

    def test_functional_update_changes_fast_state_not_slow_weights(self) -> None:
        slow_before = {name: parameter.detach().clone() for name, parameter in self.layer.fast_mlp.named_parameters()}
        state = self.layer.initial_fast_weights()

        loss = self.layer.kv_loss(self.tokens, state)
        updated_state = self.layer.sgd_step(loss, state, learning_rate=0.25, create_graph=True)

        self.assertEqual(tuple(state), tuple(updated_state))
        self.assertTrue(any(not torch.allclose(state[name], updated_state[name]) for name in state))
        for name, parameter in self.layer.fast_mlp.named_parameters():
            self.assertTrue(torch.equal(parameter, slow_before[name]))

    def test_bank_uses_one_independent_state_per_transformer_layer(self) -> None:
        bank = TTTMemoryBank(num_layers=2, model_dim=6, memory_dim=3, hidden_dim=8)
        layer_tokens = (torch.randn(2, 4, 6), torch.randn(2, 4, 6))
        state = bank.initial_fast_weights()

        self.assertEqual(len(state), 2)
        self.assertEqual(bank.kv_loss(layer_tokens, state).ndim, 0)
        self.assertEqual(bank.read(1, layer_tokens[1], state[1]).shape, layer_tokens[1].shape)


if __name__ == "__main__":
    unittest.main()
