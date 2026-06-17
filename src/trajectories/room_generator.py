"""
room_generator.py
=================
Arena map generation for rectangular, circular, and triangular rooms.

Format convention (matches Wang et al. NeurIPS 2024 codebase)
-------------------------------------------------------------
- Shape  : `(width_cm + 2*padding, height_cm + 2*padding)`
- dtype  : `float64`
- Values : `0` = accessible (free space), `1` = wall / boundary
- Default padding = 5 pixels on every side.

The accessible interior for a plain rectangular room occupies rows and columns
`[padding, padding + room_dim)`, giving exactly `width_cm x height_cm`
accessible pixels.
"""
from pathlib import Path

import numpy as np

from .constants import ARENA_MAP_KEY, DEFAULT_PADDING, FREE_SPACE, WALL


# ---------------------------------------------------------------------------
# Rectangular room
# ---------------------------------------------------------------------------

def make_square_room(
    width_cm: int,
    height_cm: int = None,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    """
    Create an arena map for a rectangular room.

    Parameters
    ----------
    width_cm  : Room width in cm (1 px = 1 cm).
    height_cm : Room height in cm.  Defaults to *width_cm* (square room).
    padding   : Wall thickness in pixels on every side (default 5, matching
                the existing 100x100 and 200x200 reference files).

    Returns
    -------
    arena_map : np.ndarray, shape `(width_cm + 2*padding, height_cm + 2*padding)`,
                dtype float64.  0 = accessible, 1 = wall.
    """
    if height_cm is None:
        height_cm = width_cm
    if width_cm <= 0 or height_cm <= 0:
        raise ValueError("Room dimensions must be positive.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")

    arena = np.ones(
        (width_cm + 2 * padding, height_cm + 2 * padding),
        dtype=np.float64,
    )
    arena[padding : padding + width_cm, padding : padding + height_cm] = FREE_SPACE
    return arena


# ---------------------------------------------------------------------------
# Shape masks (applied on top of a rectangular accessible region)
# ---------------------------------------------------------------------------

def apply_circle_mask(arena_map: np.ndarray) -> np.ndarray:
    """
    Restrict the accessible region to the largest inscribed circle.

    Any pixel that is currently accessible (value 0) but falls outside the
    inscribed circle is set to 1 (wall).  The circle is centred in the
    accessible bounding box and its radius equals half the shorter side.

    Parameters
    ----------
    arena_map : Arena map produced by :func:`make_square_room`.

    Returns
    -------
    New arena map of the same shape with the circular mask applied.
    """
    arena = arena_map.copy()
    acc_idx = np.argwhere(arena == FREE_SPACE)
    if acc_idx.size == 0:
        return arena

    x_min, y_min = acc_idx.min(axis=0)
    x_max, y_max = acc_idx.max(axis=0)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    radius = min(x_max - x_min, y_max - y_min) / 2.0

    xs, ys = np.meshgrid(
        np.arange(arena.shape[0]),
        np.arange(arena.shape[1]),
        indexing='ij',
    )
    outside = (xs - cx) ** 2 + (ys - cy) ** 2 > radius ** 2
    arena[(arena == FREE_SPACE) & outside] = WALL
    return arena


def apply_triangle_mask(
    arena_map: np.ndarray,
    kind: str = 'right_isoceles',
) -> np.ndarray:
    """
    Restrict the accessible region to a triangular shape.

    Any pixel that is currently accessible but falls outside the specified
    triangle is set to 1 (wall).  Both variants are defined in the normalised
    coordinate frame `[0, 1] x [0, 1]` of the accessible bounding box.

    Parameters
    ----------
    arena_map : Arena map produced by :func:`make_square_room`.
    kind      : Triangle variant.

                `'right_isoceles'` (default) -- right-angle at `(x_min, y_min)`
                with legs pointing in the +x and +y directions, hypotenuse
                connecting `(x_max, y_min)` and `(x_min, y_max)`.
                Interior condition: `xn + yn ≤ 1`.

                `'equilateral'` -- isoceles triangle with base along y_min,
                apex at `(center_x, y_max)`, symmetric about the vertical
                mid-axis.  Interior condition: `yn ≤ 2·xn` and `yn ≤ 2·(1-xn)`.
                (Stretched to fill the bounding box; angles are equilateral in
                normalised space.)

    Returns
    -------
    New arena map of the same shape with the triangular mask applied.
    """
    arena = arena_map.copy()
    acc_idx = np.argwhere(arena == FREE_SPACE)
    if acc_idx.size == 0:
        return arena

    x_min, y_min = acc_idx.min(axis=0)
    x_max, y_max = acc_idx.max(axis=0)
    W = float(x_max - x_min) or 1.0   # guard against degenerate 1-pixel room
    H = float(y_max - y_min) or 1.0

    xs, ys = np.meshgrid(
        np.arange(arena.shape[0]),
        np.arange(arena.shape[1]),
        indexing='ij',
    )
    xn = (xs - x_min) / W   # normalised x in [0, 1]
    yn = (ys - y_min) / H   # normalised y in [0, 1]

    if kind == 'right_isoceles':
        # Vertices (normalised): (0,0), (1,0), (0,1).
        inside = (xn >= 0) & (yn >= 0) & (xn + yn <= 1)

    elif kind == 'equilateral':
        # Vertices (normalised): (0,0), (1,0), (0.5,1).
        # Left edge  (0,0)->(0.5,1) : xn ≥ 0.5·yn
        # Right edge (1,0)->(0.5,1) : xn ≤ 1 − 0.5·yn
        inside = (yn >= 0) & (xn >= 0.5 * yn) & (xn <= 1.0 - 0.5 * yn)

    else:
        raise ValueError(
            f"Unknown triangle kind '{kind}'. "
            "Valid options: 'right_isoceles', 'equilateral'."
        )

    arena[(arena == FREE_SPACE) & ~inside] = WALL
    return arena


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_arena_map(arena_map: np.ndarray, output_dir) -> Path:
    """
    Save an arena map as `arena_map.npz` inside *output_dir*.

    The file is compressed (matching the size of the reference files) and
    stores the array under the key `'arena_map'`.

    Parameters
    ----------
    arena_map  : np.ndarray, dtype float64, values 0/1.
    output_dir : Directory path.  Created recursively if it does not exist.

    Returns
    -------
    Path to the saved `.npz` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'arena_map.npz'
    np.savez_compressed(path, **{ARENA_MAP_KEY: arena_map})
    return path
