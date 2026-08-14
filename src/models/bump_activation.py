"""Shared asymmetric sigmoid-product bump activation for RAE hidden states."""

import torch
import torch.nn as nn


class BumpActivation(nn.Module):
    """
    Learnable (per hidden unit):
        A : peak amplitude
        m : peak location

    Fixed (shared across units):
        nl : left steepness
        nr : right steepness
        s  : left saturation
        sr : right saturation, computed at initialization

    Guarantees:
        f(m) = max_x f(x) = A
    """

    def __init__(self, hidden_size: int, nl: float, nr: float, s: float = 3.0) -> None:
        super().__init__()

        self.A = nn.Parameter(torch.empty(hidden_size).uniform_(1.0, 4.0))
        self.m = nn.Parameter(torch.empty(hidden_size).uniform_(1.5, 2.5))

        nl_t = torch.tensor(float(nl))
        nr_t = torch.tensor(float(nr))
        s_t = torch.tensor(float(s))

        self.register_buffer("nl", nl_t)
        self.register_buffer("nr", nr_t)
        self.register_buffer("s", s_t)

        sr = torch.log(nr_t * (1.0 + torch.exp(s_t)) / nl_t - 1.0)
        self.register_buffer("sr", sr)

        norm = torch.sigmoid(s_t) * torch.sigmoid(sr)
        self.register_buffer("norm", norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x - self.m

        left = torch.sigmoid(self.s + self.nl * z)
        right = torch.sigmoid(self.sr - self.nr * z)

        return self.A * left * right / self.norm
