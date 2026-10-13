"""Layer-wise test-time training memory used by WAM-TTT.

The module is deliberately independent from the LDA action header so its
functional fast-weight update can be exercised on CPU.  Slow projections are
ordinary module parameters; only the MLP state passed to ``read`` and
``kv_loss`` is updated during adaptation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

FastWeightState = OrderedDict[str, Tensor]
FastWeightStates = tuple[FastWeightState, ...]


class TTTMemoryLayer(nn.Module):
    """One paper-style slow-projection / fast-MLP memory layer.

    ``fast_mlp`` is the meta-learned initial state :math:`W_0`.  The caller
    supplies an explicit functional state so updating it cannot mutate the
    module's slow parameters or leak an adaptation between episodes.
    """

    def __init__(self, model_dim: int, memory_dim: int = 48, hidden_dim: int = 128) -> None:
        super().__init__()
        if min(model_dim, memory_dim, hidden_dim) <= 0:
            raise ValueError("model_dim, memory_dim, and hidden_dim must be positive")

        self.query = nn.Linear(model_dim, memory_dim, bias=False)
        self.key = nn.Linear(model_dim, memory_dim, bias=False)
        self.value = nn.Linear(model_dim, memory_dim, bias=False)
        self.output = nn.Linear(memory_dim, model_dim, bias=False)
        self.fast_mlp = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def initial_fast_weights(self, *, detach: bool = False) -> FastWeightState:
        """Return a named fast-weight state, optionally detached for inference."""
        if detach:
            return OrderedDict(
                (name, parameter.detach().clone().requires_grad_(True))
                for name, parameter in self.fast_mlp.named_parameters()
            )
        return OrderedDict(self.fast_mlp.named_parameters())

    def read(self, tokens: Tensor, fast_weights: Mapping[str, Tensor]) -> Tensor:
        """Read a residual from visual tokens using the supplied fast state."""
        query = self.query(tokens)
        memory = functional_call(self.fast_mlp, fast_weights, (query,))
        return self.output(memory)

    def kv_loss(self, tokens: Tensor, fast_weights: Mapping[str, Tensor]) -> Tensor:
        """The K/V memory objective from WAM-TTT Equation 3."""
        predicted_values = functional_call(self.fast_mlp, fast_weights, (self.key(tokens),))
        return F.mse_loss(predicted_values, self.value(tokens))

    @staticmethod
    def sgd_step(
        loss: Tensor,
        fast_weights: Mapping[str, Tensor],
        *,
        learning_rate: float,
        create_graph: bool,
    ) -> FastWeightState:
        """Apply SGD to a functional state without mutating module parameters."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        names = tuple(fast_weights)
        values = tuple(fast_weights.values())
        gradients = torch.autograd.grad(
            loss,
            values,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        return OrderedDict(
            (name, weight - learning_rate * gradient)
            for name, weight, gradient in zip(names, values, gradients, strict=True)
        )


class TTTMemoryBank(nn.Module):
    """A distinct ``TTTMemoryLayer`` for every MMDiT transformer block."""

    def __init__(self, num_layers: int, model_dim: int, memory_dim: int = 48, hidden_dim: int = 128) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            TTTMemoryLayer(model_dim=model_dim, memory_dim=memory_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        )

    def initial_fast_weights(self, *, detach: bool = False) -> FastWeightStates:
        return tuple(layer.initial_fast_weights(detach=detach) for layer in self.layers)

    def read(self, layer_index: int, tokens: Tensor, fast_weights: Mapping[str, Tensor]) -> Tensor:
        return self.layers[layer_index].read(tokens, fast_weights)

    def kv_loss(self, layer_tokens: Sequence[Tensor], fast_weights: FastWeightStates) -> Tensor:
        self._validate_state(layer_tokens, fast_weights)
        return torch.stack(
            [layer.kv_loss(tokens, state) for layer, tokens, state in zip(self.layers, layer_tokens, fast_weights, strict=True)]
        ).sum()

    def sgd_step(
        self,
        loss: Tensor,
        fast_weights: FastWeightStates,
        *,
        learning_rate: float,
        create_graph: bool,
    ) -> FastWeightStates:
        """Apply a single joint SGD update to every layer's fast MLP state."""
        self._validate_state(self.layers, fast_weights)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        flat_weights = tuple(weight for state in fast_weights for weight in state.values())
        gradients = torch.autograd.grad(
            loss,
            flat_weights,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        gradient_iter = iter(gradients)
        return tuple(
            OrderedDict((name, weight - learning_rate * next(gradient_iter)) for name, weight in state.items())
            for state in fast_weights
        )

    def _validate_state(
        self, layer_tokens: Iterable[object], fast_weights: FastWeightStates
    ) -> None:
        if len(tuple(layer_tokens)) != len(self.layers) or len(fast_weights) != len(self.layers):
            raise ValueError("TTT memory requires one token tensor and fast state per transformer layer")
