"""Per-unit learnable sigmoid-like activation for RAE hidden states."""

import torch
import torch.nn.functional as F

INIT_SCALE = 4.0


class CustomLearnableActivation(torch.nn.Module):
    """f(x) = (1+ReLU(b)) / (1+exp(-(1+ReLU(a))*(x-k))), with f(0)=0.01."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.empty(hidden_size).uniform_(0.0, INIT_SCALE - 1.0))
        self.b = torch.nn.Parameter(torch.empty(hidden_size).uniform_(0.0, INIT_SCALE - 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_a = F.relu(self.a)
        relu_b = F.relu(self.b)
        amp = 1.0 + relu_b
        slope = 1.0 + relu_a
        k = torch.log(100.0 * amp - 1.0) / slope
        return amp / (1.0 + torch.exp(-slope * (x - k)))


def custom_ab_reg_loss(model: torch.nn.Module) -> torch.Tensor:
    """Independent L2 on raw a and b for each CustomLearnableActivation module."""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, CustomLearnableActivation):
            total = total + module.a.square().mean() + module.b.square().mean()
    return total
