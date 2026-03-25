# grassmann_utils.py for Gr(n,1)

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Literal, Optional



Metric = Literal["chordal", "geodesic", "euclidean"]


def normalize_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normalize each row of X to unit L2 norm.

    Parameters
    ----------
    X : array, shape (n_samples, dim)
        Input data, each row is treated as a vector.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    U : array, shape (n_samples, dim)
        Row-normalized matrix.
    """
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def grassmann_distance_matrix(
    X: np.ndarray,
    metric: Metric = "chordal",
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Compute a Grassmannian distance matrix for rank-1 subspaces.

    Each row of X is treated as a 1D subspace in R^d.
    Distances are based on the principal angle theta between subspaces.

        cos(theta_ij) = | <u_i, u_j> |,  with u_i, u_j unit length

    metric="chordal":  d_ij = sqrt(1 - cos(theta_ij)**2)  = |sin(theta_ij)|
    metric="geodesic": d_ij = theta_ij = arccos( cos(theta_ij) )
    metric="euclidean": d_ij = ||u_i - u_j||                    = sqrt(2 - 2 cos)

    Parameters
    ----------
    X : array, shape (n_samples, dim)
        Input data, each row is a vector representing a subspace.
    metric : {"chordal", "geodesic"}
        Type of Grassmann distance to compute.
    eps : float
        Numerical stability constant.

    Returns
    -------
    D : array, shape (n_samples, n_samples)
        Symmetric distance matrix with zeros on the diagonal.
    """
    U = normalize_rows(X, eps=eps)         # (n, d)
    G = U @ U.T                            # Gram matrix
    G = np.clip(np.abs(G), 0.0, 1.0)       # cos(theta_ij) in [0, 1]

    if metric == "chordal":
        # d_ij = |sin(theta_ij)| = sqrt(1 - cos^2)
        D = np.sqrt(np.maximum(0.0, 1.0 - G**2))

    elif metric == "geodesic":
        # d_ij = theta_ij = arccos(cos)
        D = np.arccos(G)

    elif metric == "euclidean":
        # rank-1 subspaces, orientation-free:
        # cos(theta_ij) = |<u_i, u_j>|
        # ||u_i - u_j||^2 = 2 - 2 cos(theta_ij)
        D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * G))

    else:
        raise ValueError(
            f"Unknown metric '{metric}', use 'chordal', 'geodesic', or 'euclidean'."
        )

    np.fill_diagonal(D, 0.0)
    return D


def save_distance_matrix(D: np.ndarray, path: str | Path) -> None:
    """
    Save a distance matrix to disk as .npy (float64).

    Parameters
    ----------
    D : array, shape (n_samples, n_samples)
        Distance matrix.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(D, dtype=float))


def load_distance_matrix(path: str | Path) -> np.ndarray:
    """
    Load a distance matrix saved with `save_distance_matrix`.

    Parameters
    ----------
    path : str or Path
        File path.

    Returns
    -------
    D : array
        Loaded distance matrix.
    """
    return np.load(Path(path))


def get_or_compute_grassmann_distances(
    X: np.ndarray,
    metric: Metric = "chordal",
    cache_path: Optional[str | Path] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Convenience helper: compute a Grassmann distance matrix, optionally caching it.

    Parameters
    ----------
    X : array, shape (n_samples, dim)
        Input data.
    metric : {"chordal", "geodesic"}
        Distance type.
    cache_path : str or Path, optional
        If given, will try to load D from this file.
        If it does not exist, compute and save to this path.
    force_recompute : bool
        If True, ignore any existing cache and recompute.

    Returns
    -------
    D : array, shape (n_samples, n_samples)
        Grassmann distance matrix.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file() and not force_recompute:
            return load_distance_matrix(cache_path)

    D = grassmann_distance_matrix(X, metric=metric)

    if cache_path is not None:
        save_distance_matrix(D, cache_path)

    return D