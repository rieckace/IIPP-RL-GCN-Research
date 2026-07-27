"""
layers.py

Shared encoding layers and utility functions used across DQN and GNN models.
"""

import torch
import numpy as np


def one_hot_encode_grid(
    obs: np.ndarray,
    num_classes: int = 8,
) -> torch.Tensor:
    """Convert a raw integer grid observation into a one-hot float tensor.

    The environment returns a flat vector of CellType integers (0–7).
    Neural networks learn better from one-hot encodings because they
    avoid implying ordinal relationships between cell types.

    Args:
        obs:         1-D numpy array of shape (num_cells,) with integer values.
        num_classes: Number of cell types (default 8, matching CellType enum).

    Returns:
        1-D torch float tensor of shape (num_cells * num_classes,).
        For a 10×10 grid: (100 * 8) = 800 dimensions.
    """
    obs_int = obs.astype(np.int64)
    one_hot = np.eye(num_classes, dtype=np.float32)[obs_int]  # (num_cells, num_classes)
    return torch.from_numpy(one_hot.flatten())


def batch_one_hot_encode(
    obs_batch: np.ndarray,
    num_classes: int = 8,
) -> torch.Tensor:
    """One-hot encode a batch of observations.

    Args:
        obs_batch:   2-D numpy array of shape (batch_size, num_cells).
        num_classes: Number of cell types.

    Returns:
        Float tensor of shape (batch_size, num_cells * num_classes).
    """
    batch_size, num_cells = obs_batch.shape
    obs_int = obs_batch.astype(np.int64)
    one_hot = np.eye(num_classes, dtype=np.float32)[obs_int]  # (batch, cells, classes)
    return torch.from_numpy(one_hot.reshape(batch_size, -1))
