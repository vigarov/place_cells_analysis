import os

import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from nn4n.nn import (
    RNN,
    LeakyLinearLayer,
    RecurrentLayer,
    LinearLayer,
)

from utils import RatemapAggregator


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


def masking(x, mask_rate, *, generator: torch.Generator | None = None):
    """Bernoulli input masking; optional CPU ``generator`` for reproducible draws."""
    batch_size, timesteps, _ = x.shape
    probs = torch.full((batch_size, timesteps), 1 - mask_rate)
    if generator is None:
        mask = torch.bernoulli(probs)
    else:
        mask = torch.bernoulli(probs, generator=generator)
    return x * mask.unsqueeze(2).to(x.device)


def fr_loss(states):
    mean_fr = torch.mean(states, dim=(0, 1))
    return torch.pow(mean_fr, 2).mean()


def advance_mask_generator_for_ratemap(
    traj_coord,
    wsm,
    device,
    mask_rate,
    step_size,
    n_test=None,
    *,
    mask_generator: torch.Generator | None = None,
) -> None:
    """
    Advance ``mask_generator`` by the Bernoulli draws that :func:`compute_ratemap`
    would use, without running the RAE or aggregating a rate map.
    """
    if n_test is None:
        n_test = min(500, traj_coord.shape[1] // step_size)
    with torch.no_grad():
        for step in range(n_test):
            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32, device=device)
            masking(gt_res[:, :-1], mask_rate, generator=mask_generator)


def compute_ratemap(
    rae,
    traj_coord,
    wsm,
    arena_map,
    device,
    mask_rate,
    step_size,
    n_test=None,
    *,
    mask_generator: torch.Generator | None = None,
    show_progress=False,
):
    """Aggregate RAE hidden-state rate maps over trajectory segments."""
    rm_agg = RatemapAggregator(arena_map=arena_map, device=device)
    if n_test is None:
        n_test = min(500, traj_coord.shape[1] // step_size)
    steps = range(n_test)
    if show_progress:
        steps = tqdm(steps, desc="Rate map", leave=False)
    with torch.no_grad():
        for step in steps:
            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32).to(device)
            masked_res = masking(gt_res[:, :-1], mask_rate, generator=mask_generator)
            _, states = rae(masked_res)
            rm_agg.update(coords=tc[:, 1:], states=states[0])
    return rm_agg.get_ratemap().cpu().numpy()


def train_rae(
    rae,
    traj_coord,
    wsm,
    arena_map,
    device,
    *,
    mask_rate,
    learning_rate,
    n_epochs,
    step_size,
    lambda_mse,
    lambda_fr,
    ckpt_dir=None,
    log_every=100,
):
    """
    Train the RAE and log rate maps at the end of each epoch.

    If ``ckpt_dir/latest.pth`` already exists, load it and skip training.

    Returns
    -------
    list[np.ndarray]
        Rate maps after the initial (untrained) state and after each epoch,
        or a single final rate map when loading from checkpoint.
    """
    if ckpt_dir is not None:
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        if os.path.isfile(latest_path):
            rae.load_state_dict(torch.load(latest_path, map_location=device)["rae"])
            rm = compute_ratemap(
                rae,
                traj_coord,
                wsm,
                arena_map,
                device,
                mask_rate,
                step_size,
                show_progress=True,
            )
            tqdm.write(f"Loaded {latest_path}. Skipping training.")
            return [rm]
        os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = torch.optim.Adam(rae.parameters(), lr=learning_rate)
    epoch_ratemaps = [
        compute_ratemap(
            rae, traj_coord, wsm, arena_map, device, mask_rate, step_size, show_progress=True
        )
    ]

    n_steps = traj_coord.shape[1] // step_size
    for epoch in tqdm(range(n_epochs), desc="Training"):
        for step in range(n_steps):
            optimizer.zero_grad()

            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32).to(device)
            masked_res = masking(gt_res[:, :-1], mask_rate)
            gt_res = gt_res[:, 1:]

            pred, states = rae(masked_res)
            l_mse = F.mse_loss(pred, gt_res) * lambda_mse
            l_fr = fr_loss(states[0]) * lambda_fr
            loss = l_mse + l_fr
            loss.backward()
            optimizer.step()

            if step % log_every == 0 and step != 0:
                tqdm.write(
                    f"  epoch {epoch} step {step}/{n_steps}, "
                    f"MSE: {l_mse.item():.4f}, FR: {l_fr.item():.4f}"
                )

        if ckpt_dir is not None:
            torch.save({"rae": rae.state_dict()}, os.path.join(ckpt_dir, "latest.pth"))
            torch.save({"rae": rae.state_dict()}, os.path.join(ckpt_dir, f"rae_{epoch}.pth"))

        rm = compute_ratemap(rae, traj_coord, wsm, arena_map, device, mask_rate, step_size)
        epoch_ratemaps.append(rm)
        tqdm.write(
            f"  epoch {epoch} ratemaps: shape={rm.shape}, "
            f"mean peak activation={np.nanmax(rm, axis=(1, 2)).mean():.4f}"
        )

    tqdm.write(f"Done. Logged {len(epoch_ratemaps)} epoch ratemaps.")
    return epoch_ratemaps
