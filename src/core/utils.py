from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

MaskMethod = Literal["timestep", "cell"]


def masking_timesteps(x, mask_rate, *, generator: torch.Generator | None = None):
    """Zero whole timesteps; `mask_rate` = probability each timestep is zeroed."""
    batch_size, timesteps, _ = x.shape
    probs = torch.full((batch_size, timesteps), 1 - mask_rate)
    if generator is None:
        mask = torch.bernoulli(probs)
    else:
        mask = torch.bernoulli(probs, generator=generator)
    return x * mask.unsqueeze(2).to(x.device)


def masking_cells(x, mask_rate, *, generator: torch.Generator | None = None):
    """Zero individual input cells; `mask_rate` = probability each cell is zeroed."""
    keep_prob = 1.0 - mask_rate
    probs = torch.full(x.shape, keep_prob)
    if generator is None:
        mask = torch.bernoulli(probs)
    else:
        mask = torch.bernoulli(probs, generator=generator)
    return x * mask.to(x.device)


def apply_input_mask(
    x: torch.Tensor,
    mask_rate: float,
    *,
    mask_method: MaskMethod = "timestep",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if mask_method == "timestep":
        return masking_timesteps(x, mask_rate, generator=generator)
    if mask_method == "cell":
        return masking_cells(x, mask_rate, generator=generator)
    raise ValueError(f"mask_method must be 'timestep' or 'cell', got {mask_method!r}")


def visualize_response_map(response_map, random_cell=False, im_width=3, n_cols=5, n_rows=2):
    """ Visualize the response map """
    n_cells = response_map.shape[0]
    if random_cell:
        cell_indices = np.random.choice(n_cells, size=n_cols*n_rows, replace=False)
    else:
        cell_indices = range(n_cells)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(im_width*n_cols, im_width*n_rows))
    # Display colormap range using the global min and max across sampled cells
    plotted_cells = [response_map[idx] for idx in cell_indices[:n_cols*n_rows]]
    all_vals = np.concatenate([cell.flatten() for cell in plotted_cells])
    vmin, vmax = 0, np.nanmax(all_vals)

    im = None
    for i, ax in enumerate(axes.flat):
        if i < len(cell_indices):
            rm = response_map[cell_indices[i]]
            im = ax.imshow(rm, cmap='jet', vmin=vmin, vmax=vmax)
            min_fr, max_fr, mean_fr = np.nanmin(rm), np.nanmax(rm), np.nanmean(rm)
            ax.set_title(f'min: {min_fr:.2f}, max: {max_fr:.2f}\nmean: {mean_fr:.2f}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    if im is not None:
        # Add a colorbar on the right side (vertical, middle position)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Firing rate", orientation="vertical", location="right")
        # fig.subplots_adjust(right=2.8)

        
    # plt.tight_layout()

    plt.show()

class RatemapAggregator:
    def __init__(self, arena_map, device=None):
        """
        Class to accumulate partial data for rate-map computation.

        Parameters
        ----------
        arena_map : torch.Tensor or np.ndarray
            Map of shape (n_x, n_y), 0 for free space, 1 for walls, 
            used for figuring out dimensions and for masking.
        device : str or torch.device, optional
            The device on which to store the data (CPU or GPU). 
            If None, uses arena_map device if it's a torch.Tensor, 
            otherwise "cpu".
        """
        # If arena_map is numpy array, convert to torch
        if isinstance(arena_map, np.ndarray):
            arena_map = torch.as_tensor(arena_map)
        
        self.arena_map = arena_map
        self.dims = arena_map.shape  # (n_x, n_y)
        self.n_cells = None
        # Infer device
        if device is None:
            self.device = arena_map.device if arena_map.is_cuda else torch.device('cpu')
        else:
            self.device = torch.device(device)

    
    def init_counts(self):
        self.partial_sums = torch.zeros(
            (self.n_cells, *self.dims),
            dtype=torch.float32,
            device=self.device
        )
        self.finite_counts = torch.zeros(
            (self.n_cells, *self.dims),
            dtype=torch.float32,
            device=self.device
        )
        # shape: (n_x, n_y)
        self.visit_counts = torch.zeros(
            self.dims,
            dtype=torch.float32,
            device=self.device
        )

    def update(self, states, coords):
        """
        Accumulate partial sums and visit counts from new data.

        Parameters
        ----------
        states : torch.Tensor or np.ndarray
            Shape=(n_batches, n_timesteps, n_cells) or (n_timesteps, n_batches, n_cells).
        coords : torch.Tensor or np.ndarray
            Shape=(n_batches, n_timesteps, 2) or (n_timesteps, n_batches, 2).
        """
        if self.n_cells is None:
            self.n_cells = states.shape[-1]
            self.init_counts()

        # Convert to torch if numpy
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states)
        if isinstance(coords, np.ndarray):
            coords = torch.as_tensor(coords)

        # Move to the same device
        states = states.float().to(self.device)
        coords = coords.float().to(self.device)

        # Ensure correct dtype
        states = states.float()
        coords = torch.round(coords).long()  # round and convert to long

        # Standardize shapes
        assert states.dim() == 3, "states must have 3 dims: (n_batches, n_timesteps, n_cells)"
        assert coords.dim() == 3, "coords must have 3 dims: (n_batches, n_timesteps, 2)"

        # Flatten
        coords = coords.reshape(-1, 2)     # (n_batches * n_timesteps, 2)
        states = states.reshape(-1, self.n_cells)  # (n_batches * n_timesteps, n_cells)

        # Flatten partial sums and visit_counts for fast index_add
        flat_sums = self.partial_sums.view(self.n_cells, -1)  # shape: (n_cells, n_x*n_y)
        flat_finite = self.finite_counts.view(self.n_cells, -1)
        flat_counts = self.visit_counts.view(-1)              # shape: (n_x*n_y)

        # Convert (row, col) coords into linear indices
        dims = self.dims
        flat_coords = coords[:, 0] * dims[1] + coords[:, 1]  # shape: (n_samples,)

        # Ignore non-finite states (NaN + inf) so one bad sample cannot poison a bin.
        finite = torch.isfinite(states)
        safe_states = torch.where(finite, states, torch.zeros_like(states))

        # Accumulate partial sums
        # states.T shape: (n_cells, n_samples)
        # so we add states.T to the flat_sums at the flattened coordinate indices
        flat_sums.index_add_(1, flat_coords, safe_states.T)
        flat_finite.index_add_(1, flat_coords, finite.T.float())

        # Accumulate visit counts
        flat_counts.index_add_(
            0, 
            flat_coords, 
            torch.ones_like(flat_coords, dtype=torch.float32)
        )

    def get_ratemap(self):
        """
        Returns the final normalized firing fields (n_cells, n_x, n_y).
        Unvisited points (visit_count=0) will be NaN.
        """
        denom = self.finite_counts.clamp(min=1.0)
        ratemap = self.partial_sums / denom

        # Bins with no finite samples for a unit are NaN.
        ratemap[self.finite_counts == 0] = float('nan')

        return ratemap

    def reset(self):
        """
        Reset the aggregator (clears all partial sums and counts).
        """
        self.partial_sums.zero_()
        self.finite_counts.zero_()
        self.visit_counts.zero_()


def advance_mask_generator_for_ratemap(
    traj_coord,
    wsm,
    device,
    mask_rate,
    step_size,
    n_test=None,
    *,
    mask_method: MaskMethod = "timestep",
    mask_generator: torch.Generator | None = None,
) -> None:
    """
    Advance `mask_generator` by the Bernoulli draws that `compute_ratemap`
    would use, without running the RAE or aggregating a rate map.
    """
    if n_test is None:
        n_test = min(500, traj_coord.shape[1] // step_size)
    with torch.no_grad():
        for step in range(n_test):
            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32, device=device)
            apply_input_mask(
                gt_res[:, :-1],
                mask_rate,
                mask_method=mask_method,
                generator=mask_generator,
            )


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
    mask_method: MaskMethod = "timestep",
    mask_generator: torch.Generator | None = None,
    show_progress=False,
    return_mse: bool = False,
):
    """Aggregate RAE hidden-state rate maps over trajectory segments.

    When `return_mse` is True, also return the mean unweighted MSE between
    model predictions and ground-truth responses on the same segments.
    """
    rm_agg = RatemapAggregator(arena_map=arena_map, device=device)
    max_steps = traj_coord.shape[1] // step_size
    if n_test is None:
        n_test = min(500, max_steps)
    else:
        n_test = min(n_test, max_steps)
    steps = range(n_test)
    if show_progress:
        steps = tqdm(steps, desc="Rate map", leave=False)
    mse_values: list[float] = []
    with torch.no_grad():
        init_states = None
        for step in steps:
            tc = traj_coord[:, step * step_size : (step + 1) * step_size]
            gt_res = torch.as_tensor(wsm.get_response(tc), dtype=torch.float32).to(device)
            masked_res = apply_input_mask(
                gt_res[:, :-1],
                mask_rate,
                mask_method=mask_method,
                generator=mask_generator,
            )
            gt_res_target = gt_res[:, 1:]
            pred, states = rae(masked_res, init_states=init_states)
            if return_mse:
                mse_values.append(float(F.mse_loss(pred, gt_res_target).item()))
            init_states = [states[0][:, -1, :].clone()]
            rm_agg.update(coords=tc[:, 1:], states=states[0])
    ratemap = rm_agg.get_ratemap().cpu().numpy()
    if return_mse:
        mse = float(np.mean(mse_values)) if mse_values else float("nan")
        return ratemap, mse
    return ratemap
