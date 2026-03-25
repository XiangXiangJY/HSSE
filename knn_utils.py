# knn_utils.py

import numpy as np


def knn_indices_from_distance(D, k, include_self=True):
    """
    For each point i, return indices of its k nearest neighbors (by distance).

    Parameters
    ----------
    D : array-like, shape (n, n)
        Symmetric distance matrix.
    k : int
        Number of neighbors to return per point.
    include_self : bool
        If True, the first neighbor is 'i' itself (distance 0).
        If False, self is excluded and the k closest *other* points are returned.

    Returns
    -------
    knn_idx : ndarray, shape (n, k)
        knn_idx[i] is the list of neighbor indices for point i.
    """
    D = np.asarray(D)
    n = D.shape[0]

    # argsort along each row: smallest distance first
    order = np.argsort(D, axis=1)

    if include_self:
        knn_idx = order[:, :k]
    else:
        # skip index 0 in sorted order, which should be i itself
        knn_idx = order[:, 1 : (k + 1)]

    return knn_idx


def get_local_patch(X, D, center, k, include_self=True):
    """
    Extract a local patch around a single center cell using kNN.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Global feature matrix (e.g. UMAP or PCA embedding).
    D : array-like, shape (n, n)
        Global distance matrix (e.g. Grassmann distance).
    center : int
        Index of the center cell.
    k : int
        Number of neighbors to include in the patch.
    include_self : bool
        Whether to force `center` itself into the patch.

    Returns
    -------
    idx : ndarray, shape (m,)
        Indices of cells in this local patch.
    X_local : ndarray, shape (m, d)
        Features restricted to these indices.
    D_local : ndarray, shape (m, m)
        Distance matrix restricted to these indices (same order as idx).
    center_local_index : int
        The index in {0,...,m-1} where the center cell appears in X_local/D_local.
    
    """
    X = np.asarray(X)
    D = np.asarray(D)

    knn_idx = knn_indices_from_distance(D, k=k, include_self=include_self)
    idx = knn_idx[center]

    X_local = X[idx]
    D_local = D[np.ix_(idx, idx)]
    # Find where "center" is inside the local patch
    center_local_index = np.where(idx == center)[0][0]

    return idx, X_local, D_local, center_local_index