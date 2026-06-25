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

    We ONLY encode a binary label in {0,1} to be used in the restriction map:

      - label = 0 for the center cell
      - label = 1 for all other cells

    Parameters
    ----------
    X : array-like, shape (m, d)
        Local feature matrix for the patch.
    center_index : int
        The index in [0..m-1] corresponding to the center cell.

    Returns
    -------
    extra_data : dict
        Keys are (i,) for vertex i.
        Values are dicts with field:
          'label'  : float in {0.0, 1.0}
    """
    X = np.asarray(X)
    m = X.shape[0]

    extra_data = {}
    for i in range(m):
        if i == center_index:
            extra_data[(i,)] = {"label": 0.0}   # center
        else:
            extra_data[(i,)] = {"label": 1.0}   # neighbor

    return extra_data


def make_restriction(D, sigma=None, alpha=0.0):
    """
    Build a scalar restriction function using the local Grassmann distance D
    and (optionally) vertex labels stored in sst.extra_data[(i,)]['label'] in {0,1}.

    Geometry part (same as your best-performing model):
      k(d) = exp( - d^2 / sigma^2 )

    Label modulation:
      For vertices i,j with labels l_i, l_j in {0,1},
      define m_ij = |l_i - l_j| in {0,1}.

      We use a soft penalty
          exp(-alpha * m_ij),
      so that:
        - if l_i = l_j (same label), m_ij = 0, factor = 1
        - if l_i != l_j (different labels), m_ij = 1, factor = exp(-alpha)

    alpha controls the strength of label information:
      - alpha = 0.0 : purely geometric PSL (your current best model)
      - small alpha > 0 : mild suppression of cross-label restrictions
      - larger alpha    : stronger label influence

    Rules
    -----
    vertex -> edge:
        rho_{i -> (i,j)} =
            exp( - d(i,j)^2 / sigma^2 ) * exp( - alpha * m_ij )

    edge   -> triangle:
        rho_{(i,j) -> (i,j,k)} =
            0.5 * [
                exp( - d(i,k)^2 / sigma^2 ) * exp( - alpha * m_ik )
              + exp( - d(j,k)^2 / sigma^2 ) * exp( - alpha * m_jk )
            ]

    Parameters
    ----------
    D : array-like, shape (m, m)
        Local distance matrix.
    sigma : float or None
        Scale of the exponential kernel. If None, uses median of non-zero distances.
    alpha : float, default 0.0
        Strength of label information. alpha = 0 recovers the geometry-only model.

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
            k_ij = np.exp(-(d_ij ** 2) / (sigma ** 2))

            # labels in {0,1}
            li = sst.extra_data[(i,)]['label']
            lj = sst.extra_data[(j,)]['label']
            m_ij = abs(li - lj)   # 0 if same, 1 if different

            return float(k_ij * np.exp(-alpha * m_ij))

        # edge -> triangle
        if len(simplex) == 2 and len(coface) == 3:
            i, j = simplex
            # the third vertex
            k = [v for v in coface if v not in simplex][0]

            d_ik = D[i, k]
            d_jk = D[j, k]

            k_ik = np.exp(-(d_ik ** 2) / (sigma ** 2))
            k_jk = np.exp(-(d_jk ** 2) / (sigma ** 2))

            li = sst.extra_data[(i,)]['label']
            lj = sst.extra_data[(j,)]['label']
            lk = sst.extra_data[(k,)]['label']

            m_ik = abs(li - lk)
            m_jk = abs(lj - lk)

            val = 0.5 * (
                k_ik * np.exp(-alpha * m_ik) +
                k_jk * np.exp(-alpha * m_jk)
            )
            return float(val)

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
    alpha=0.0,
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
    center_index : int, default 0
        Index of the center cell in the local patch.
    alpha : float, default 0.0
        Strength of label modulation in the restriction maps.

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
    restriction = make_restriction(D_local, sigma=sigma, alpha=alpha)

    # Build sheaf and PSL
    sst = sheaf_simplex_tree(st, extra_data, restriction)
    psl = PersistentSheafLaplacian(sst)

    # Compute spectra for requested dimensions
    results = {}
    for dim in dims:
        results[dim] = psl.spectra(dim=dim, a=a, b=b)

    return results