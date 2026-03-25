# psl_utils.py

import numpy as np
import gudhi
from petls import sheaf_simplex_tree, PersistentSheafLaplacian


def build_simplex_tree_from_dist(D, r_max=None, max_dim=2):
    """
    Build a Gudhi Rips complex from a local distance matrix D.

    Parameters
    ----------
    D : array-like, shape (m, m)
        Symmetric distance matrix for a local patch.
    r_max : float or None
        Max edge length for Rips complex. If None, uses max non-zero distance.
    max_dim : int
        Max simplex dimension for the Rips complex.

    Returns
    -------
    st : gudhi.SimplexTree
    r_used : float
        The radius actually used as max_edge_length.
    """
    D = np.asarray(D)
    dists = D[D > 0]

    if r_max is None:
        r_used = float(dists.max()) if dists.size > 0 else 1.0
    else:
        r_used = float(r_max)

    rips = gudhi.RipsComplex(distance_matrix=D.tolist(), max_edge_length=r_used)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    return st, r_used

def build_extra_data(X, center_index=0):
    """
    Attach per-vertex data to the sheaf_simplex_tree.

    We assign:
      - charge = +1 for the center cell
      - charge = -1.5 for all other cells

    Parameters
    ----------
    X : array-like, shape (m, d)
        Local feature matrix for the patch.
    center_index : int
        The index in [0..m-1] corresponding to the center cell.

    Returns
    -------
    extra_data : dict
        Keys are (i,) for vertex i. Values are dicts with 'charge'.
    """
    X = np.asarray(X)
    m = X.shape[0]

    extra_data = {}
    for i in range(m):
        if i == center_index:
            extra_data[(i,)] = {"charge": 1.0}
        else:
            extra_data[(i,)] = {"charge": -1.5}

    return extra_data


def make_restriction(D, sigma=None):
    """
    Build a scalar restriction function using the local Grassmann distance D.

    Rules
    -----
    vertex -> edge:  rho_{i -> (i,j)} = exp( - d(i,j)^2 / sigma^2 )
    edge   -> triangle: average of weights toward the third vertex.

    Parameters
    ----------
    D : array-like, shape (m, m)
        Local distance matrix.
    sigma : float or None
        Scale of the exponential kernel. If None, uses median of non-zero distances.

    Returns
    -------
    my_restriction : callable
        Function (simplex, coface, sst) -> float
        to be passed into petls.sheaf_simplex_tree.
    """
    D = np.asarray(D)
    dists = D[D > 0]

    if sigma is None:
        sigma = float(np.median(dists)) if dists.size > 0 else 1.0

    def my_restriction(simplex, coface, sst):
        # vertex -> edge
        if len(simplex) == 1 and len(coface) == 2:
            i = simplex[0]
            j = coface[0] if coface[1] == i else coface[1]
            d_ij = D[i, j]
            w_ij = np.exp(-(d_ij ** 2) / (sigma ** 2))
            return float(w_ij)

        # edge -> triangle
        if len(simplex) == 2 and len(coface) == 3:
            i, j = simplex
            k = [v for v in coface if v not in simplex][0]

            d_ik = D[i, k]
            d_jk = D[j, k]

            w_ik = np.exp(-(d_ik ** 2) / (sigma ** 2))
            w_jk = np.exp(-(d_jk ** 2) / (sigma ** 2))

            return float(0.5 * (w_ik + w_jk))

        # for other codimensions (not used here)
        return 1.0

    return my_restriction


def compute_psl_eigs(
    X_local,
    D_local,
    a=0.0,
    b=None,
    max_dim=2,
    sigma=None,
    dims=(0, 1, 2),
    center_index=0, 
):
    """
    High-level helper:
    given a local patch (X_local, D_local), build a PSL and compute spectra.

    Parameters
    ----------
    X_local : array-like, shape (m, d)
        Local features for the m cells in this patch.
    D_local : array-like, shape (m, m)
        Local Grassmann distance matrix for these cells.
    a, b : float
        Filtration interval [a, b]. If b is None, b = max edge length in D_local.
    max_dim : int
        Max simplex dimension for the Rips complex.
    sigma : float or None
        Kernel scale for the restriction map.
    dims : iterable of int
        Which homological dimensions to compute spectra for.

    Returns
    -------
    spectra : dict
        Keys are dimensions in `dims`.
        Values are lists / arrays of eigenvalues from psl.spectra().
    """
    # Build simplicial complex
    st, r_used = build_simplex_tree_from_dist(D_local, r_max=None, max_dim=max_dim)

    # If no explicit upper bound was given, use r_used
    if b is None:
        b = r_used

    # Attach vertex data and restriction function
    extra_data = build_extra_data(X_local, center_index=center_index)
    restriction = make_restriction(D_local, sigma=sigma)

    # Build sheaf and PSL
    sst = sheaf_simplex_tree(st, extra_data, restriction)
    psl = PersistentSheafLaplacian(sst)

    # Compute spectra for requested dimensions
    results = {}
    for dim in dims:
        results[dim] = psl.spectra(dim=dim, a=a, b=b)

    return results