"""
Parts of this file have been adapted from
https://github.com/nicola-decao/diffmask/blob/master/diffmask/models/gates.py
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from utils.distributions import RectifiedStreched, BinaryConcrete


class MLPGate(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        This is an MLP with the following structure;
        Linear(input_size, hidden_size), Tanh(), Linear(hidden_size, 1)
        The bias of the last layer is set to 5.0 to start with high probability
        of keeping states (fundamental for good convergence as the initialized
        DiffMask has not learned what to mask yet).

        Args:
            input_size (int): the number of input features
            hidden_size (int): the number of hidden units
            bias (bool): whether to use a bias term
        """
        super().__init__()

        self.f = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, 1, bias=bias)),
        )

        if bias:
            self.f[-1].bias.data[:] = 5.0

    def forward(self, *args: Tensor) -> Tensor:
        return self.f(torch.cat(args, -1))


class MLPMaxGate(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mul_activation: float = 10.0,
        add_activation: float = 5.0,
        bias: bool = True,
    ):
        """
        This is an MLP with the following structure;
        Linear(input_size, hidden_size), Tanh(), Linear(hidden_size, 1)
        The bias of the last layer is set to 5.0 to start with high probability
        of keeping states (fundamental for good convergence as the initialized
        DiffMask has not learned what to mask yet).
        It also uses a scaler for the output of the activation function.

        Args:
            input_size (int): the number of input features
            hidden_size (int): the number of hidden units
            mul_activation (float): the scaler for the output of the activation function
            add_activation (float): the offset for the output of the activation function
            bias (bool): whether to use a bias term
        """
        super().__init__()

        self.f = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            nn.utils.weight_norm(nn.Linear(hidden_size, 1, bias=bias)),
            nn.Tanh(),
        )
        self.add_activation = nn.Parameter(torch.tensor(add_activation))
        self.mul_activation = mul_activation

    def forward(self, *args: Tensor) -> Tensor:
        return self.f(torch.cat(args, -1)) * self.mul_activation + self.add_activation


class DiffMaskGateInput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: nn.Module = MLPMaxGate,
        mul_activation: float = 10.0,
        add_activation: float = 5.0,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: Tensor = None,
    ):
        """This is a DiffMask module that masks the input of the first layer.

        Args:
            hidden_size (int): the size of the hidden representations
            hidden_attention (int) the amount of units in the gate's hidden (bottleneck) layer
            num_hidden_layers (int): the number of hidden layers (and thus gates to use)
            max_position_embeddings (int): the amount of placeholder embeddings to learn for the masked positions
            gate_fn (nn.Module): the PyTorch module to use as a gate
            mul_activation (float): the scaler for the output of the activation function
            add_activation (float): the offset for the output of the activation function
            gate_bias (bool): whether to use a bias term
            placeholder (bool): whether to use placeholder embeddings or a zero vector
            init_vector (Tensor): the initial vector to use for the placeholder embeddings
        """
        super().__init__()

        # Create a ModuleList with the gates
        self.g_hat = nn.ModuleList(
            [
                gate_fn(
                    hidden_size * 2,
                    hidden_attention,
                    mul_activation,
                    add_activation,
                    gate_bias,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            # Use a placeholder embedding for the masked positions
            self.placeholder = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty((1, max_position_embeddings, hidden_size))
                )
                if init_vector is None
                else init_vector.view(1, 1, hidden_size).repeat(
                    1, max_position_embeddings, 1
                )
            )
        else:
            # Use a zero vector for the masked positions
            self.register_buffer(
                "placeholder",
                torch.zeros((1, 1, hidden_size)),
            )

    def forward(
        self, hidden_states: tuple[Tensor], layer_pred: Optional[int], aggregated: bool = True
    ) -> tuple[tuple[Tensor], Tensor, Tensor, Tensor, Tensor]:
        # Concatenate the output of all the gates
        logits = torch.cat(
            [
                self.g_hat[i](hidden_states[0], hidden_states[i])
                for i in range(
                    (layer_pred + 1) if layer_pred is not None else len(hidden_states)
                )
            ],
            -1,
        )

        # Define a Hard Concrete distribution
        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits),
            l=-0.2,
            r=1.0,
        )

        # Calculate the expectation for the full gate probabilities
        # These act as votes for the masked positions
        gates_full = dist.rsample().cumprod(-1)
        if aggregated:
            expected_L0_full = dist.log_expected_L0().cumsum(-1)
        else:
            expected_L0_full = dist.log_expected_L0()

        # Extract the probabilities from the last layer, which acts
        # as an aggregation of the votes per position
        gates = gates_full[..., -1]
        expected_L0 = expected_L0_full[..., -1]

        return (
            hidden_states[0] * gates.unsqueeze(-1)
            + self.placeholder[:, : hidden_states[0].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )


# class DiffMaskGateHidden(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         hidden_attention: int,
#         num_hidden_layers: int,
#         max_position_embeddings: int,
#         gate_fn: nn.Module = MLPMaxGate,
#         gate_bias: bool = True,
#         placeholder: bool = False,
#         init_vector: Tensor = None,
#     ):
#         super().__init__()
#
#         self.g_hat = nn.ModuleList(
#             [
#                 gate_fn(hidden_size, hidden_attention, bias=gate_bias)
#                 for _ in range(num_hidden_layers)
#             ]
#         )
#
#         if placeholder:
#             self.placeholder = nn.ParameterList(
#                 [
#                     nn.Parameter(
#                         nn.init.xavier_normal_(
#                             torch.empty((1, max_position_embeddings, hidden_size))
#                         )
#                         if init_vector is None
#                         else init_vector.view(1, 1, hidden_size).repeat(
#                             1, max_position_embeddings, 1
#                         )
#                     )
#                     for _ in range(num_hidden_layers)
#                 ]
#             )
#         else:
#             self.register_buffer(
#                 "placeholder",
#                 torch.zeros((num_hidden_layers, 1, 1, hidden_size)),
#             )
#
#     def forward(
#         self, hidden_states: tuple[Tensor], layer_pred: Optional[int]
#     ) -> tuple[tuple[Tensor], Tensor, Tensor, Tensor, Tensor]:
#         if layer_pred is not None:
#             logits = self.g_hat[layer_pred](hidden_states[layer_pred])
#         else:
#             logits = torch.cat(
#                 [self.g_hat[i](hidden_states[i]) for i in range(len(hidden_states))], -1
#             )
#
#         dist = RectifiedStreched(
#             BinaryConcrete(torch.full_like(logits, 0.2), logits),
#             l=-0.2,
#             r=1.0,
#         )
#
#         gates_full = dist.rsample()
#         expected_L0_full = dist.log_expected_L0()
#
#         gates = gates_full if layer_pred is not None else gates_full[..., :1]
#         expected_L0 = (
#             expected_L0_full if layer_pred is not None else expected_L0_full[..., :1]
#         )
#
#         layer_pred = layer_pred or 0  # equiv to "layer_pred if layer_pred else 0"
#         return (
#             hidden_states[layer_pred] * gates
#             + self.placeholder[layer_pred][:, : hidden_states[layer_pred].shape[-2]]
#             * (1 - gates),
#             gates.squeeze(-1),
#             expected_L0.squeeze(-1),
#             gates_full,
#             expected_L0_full,
#         )
