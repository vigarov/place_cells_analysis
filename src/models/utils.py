import torch
from models.nn4n.nn import (
    RNN,
    LeakyLinearLayer,
    RecurrentLayer,
    LinearLayer,
)


def build_rae(n_cells, n_hidden, device):
    """Build and move a readout autoencoder RNN to the target device."""
    input_layer = LinearLayer(input_dim=n_cells, output_dim=n_hidden)
    output_layer = LinearLayer(input_dim=n_hidden, output_dim=n_cells)
    leaky_layer = LeakyLinearLayer(
        linear_layer=LinearLayer(input_dim=n_hidden, output_dim=n_hidden),
        activation=torch.nn.ReLU(),
        alpha=0.1,
        learn_alpha=False,
        preact_noise=0,
        postact_noise=0,
    )
    rae = RNN(
        readout_layer=output_layer,
        recurrent_layers=[
            RecurrentLayer(
                leaky_layer=leaky_layer,
                projection_layer=input_layer,
            )
        ],
    )
    return rae.to(device)
