import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import umap

from auxilary import preprocess_data, load_X, load_y
from scale import generate_scales
from grassutils import get_or_compute_grassmann_distances
from knn_utils import get_local_patch
from psl_utilsA import compute_psl_eigs


# ===================== Global parameter config =====================

DATA_NAME = "GSE67835"
DATA_PATH = "./data/"

# PCA parameters (param 1)
PCA_N_COMPONENTS = 50
PCA_RANDOM_STATE = 1

# UMAP parameters (param 2, 3, 4)
UMAP_N_COMPONENTS = 20          # param 2
UMAP_MIN_DIST = 0.4             # param 3
UMAP_METRIC = "euclidean"       # param 4

# Multi-scale neighbors (param 5)
N_SCALES = 5
SCALES_MODE = "power1.2"
SCALES_MIN_CAP = 5
SCALES_MAX_CAP = 50

# Local patch sizes (param 6)
K_LOCAL_LIST = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80]

# Filtration sampling (param 7)
N_FILTRATIONS = 10              # number of sampled intervals
FILTRATION_PATTERN = "segments_linear"  # currently: [0, 0.1b], ..., [0, b]

# PSL homology dimensions (param 8)
PSL_DIMS = (0, 1)

# Rips / PSL extra parameters
RIPS_MAX_DIM = 2
PSL_SIGMA = None                # None -> use median distance in make_restriction
NORMALIZE_METHOD = "max"        # "max" / "median" / "minmax"

# NEW: label influence strength
# alpha = 0.0  -> geometry-only PSL (original best model)
# small alpha>0 -> weak label effect
ALPHA_LABEL = 0.01               #  0.0  ，  0.01, 0.05  

OUT_DIST_ROOT = "distances"
OUT_EIGS_ROOT = "psl_eigs"

OUT_DIST_DIR = os.path.join(OUT_DIST_ROOT, DATA_NAME)
OUT_EIGS_DIR = os.path.join(OUT_EIGS_ROOT, DATA_NAME)


# ===================== Utility: argument parser =====================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--center_start",
        type=int,
        default=0,
        help="starting cell index (inclusive, 0-based)",
    )
    parser.add_argument(
        "--center_end",
        type=int,
        default=None,
        help="ending cell index (exclusive, 0-based). If None, use n_cells.",
    )
    return parser.parse_args()


# ===================== Utility: distance normalization =====================

def normalize_distance_matrix(D, method="median", clip=True, eps=1e-12):
    """
    Normalize a distance matrix D into [0,1] range (or similar),
    with several robust normalization methods.
    """
    D = np.asarray(D, dtype=float)
    D = 0.5 * (D + D.T)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    # allow no normalization
    if method == "none":
        return D

    positive_vals = D[D > 0]
    if positive_vals.size == 0:
        return D

    if method == "max":
        scale = np.max(positive_vals)
    elif method == "median":
        scale = np.median(positive_vals)
    elif method == "minmax":
        scale = np.max(positive_vals)
    else:
        raise ValueError(f"Unknown normalization method '{method}'")

    scale = max(scale, eps)
    D_norm = D / scale

    if clip:
        D_norm = np.clip(D_norm, 0.0, 1.0)

    return D_norm


# ===================== Utility: filtration intervals =====================

def generate_filtration_intervals(b_max, n_filtrations=N_FILTRATIONS, pattern=FILTRATION_PATTERN):
    """
    Generate a list of [a,b] intervals for PSL.

    pattern == "prefix_linear":
      [0, (1/n)b_max], [0, (2/n)b_max], ..., [0, b_max]

    pattern == "segments_linear":
      [0, b1], [b1, b2], ..., [b_{n-1}, b_n]
      where b_i are linearly spaced between 0 and b_max.
    """
    if b_max <= 0:
        return [(0.0, 0.0)]

    if pattern == "prefix_linear":
        fracs = np.linspace(1.0 / n_filtrations, 1.0, n_filtrations)
        intervals = [(0.0, float(f * b_max)) for f in fracs]
        return intervals

    elif pattern == "segments_linear":
        # n_filtrations intervals → n_filtrations+1 breakpoints
        edges = np.linspace(0.0, 1.0, n_filtrations + 1)
        intervals = []
        for i in range(n_filtrations):
            a = float(edges[i] * b_max)
            b = float(edges[i + 1] * b_max)
            intervals.append((a, b))
        return intervals

    else:
        raise ValueError(f"Unknown filtration pattern '{pattern}'")


# ===================== Utility: file name helpers =====================

def format_float_short(x):
    """Format float for file name, e.g. 0.2 -> '0p2'."""
    return str(x).replace(".", "p")


def format_dims(dims):
    """dims=(0,1,2) -> 'd0-1-2'."""
    return "d" + "-".join(str(int(d)) for d in dims)


def make_psl_cell_filename(
    data,
    center,
    scale,
    k_local,
    pca,
    umap_dim,
    umap_metric,
    umap_min_dist,
    norm_method,
    n_filtrations,
    pattern,
    dims,
    rips_max_dim,
    sigma,
    alpha,
    ext="npy",
):
    """
    Build a unique filename encoding all key parameters + cell index.
    """
    metric_clean = umap_metric.replace("euclidean", "euc").replace("correlation", "corr")
    md_tag = format_float_short(umap_min_dist)
    dims_tag = format_dims(dims)

    if sigma is None:
        sigma_tag = "auto"
    else:
        sigma_tag = format_float_short(sigma)

    alpha_tag = format_float_short(alpha)

    fname = (
        f"{data}"
        f"_cell{center}"
        f"_pca{pca}"
        f"_umap{umap_dim}"
        f"_met{metric_clean}"
        f"_md{md_tag}"
        f"_scale{scale}"
        f"_k{k_local}"
        f"_norm{norm_method}"
        f"_f{n_filtrations}"
        f"_pat{pattern}"
        f"_{dims_tag}"
        f"_rips{rips_max_dim}"
        f"_sig{sigma_tag}"
        f"_alpha{alpha_tag}"
        f"_psl.{ext}"
    )
    return fname


# ===================== Dimensionality reduction =====================

def run_pca(X, n_components=PCA_N_COMPONENTS, random_state=PCA_RANDOM_STATE):
    """
    PCA on X (n_samples x n_features).
    Returns X_pca (n_samples x n_components) and the fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def run_umap_with_neighbors(
    X,
    n_neighbors,
    n_components=UMAP_N_COMPONENTS,
    random_state=1,
    metric=UMAP_METRIC,
    min_dist=UMAP_MIN_DIST,
):
    """
    Run UMAP on X with a given n_neighbors.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    return embedding, reducer


def run_umap_for_scales(
    X_pca,
    scales,
    n_components=UMAP_N_COMPONENTS,
    random_state=1,
    metric=UMAP_METRIC,
    min_dist=UMAP_MIN_DIST,
):
    """
    For each s in scales, run UMAP with n_neighbors = s on X_pca.

    Returns:
        embeddings_dict: {scale: embedding (n_samples x n_components)}
    """
    embeddings = {}
    for s in scales:
        emb, _ = run_umap_with_neighbors(
            X_pca,
            n_neighbors=s,
            n_components=n_components,
            random_state=random_state,
            metric=metric,
            min_dist=min_dist,
        )
        embeddings[s] = emb
    return embeddings


# ===================== Main pipeline =====================

def main():
    args = parse_args()

    data = DATA_NAME
    data_path = DATA_PATH

    # 1. Load data
    X = load_X(data, data_path)
    y = load_y(data, data_path)
    X = np.log10(1 + X).T

    n_cells = X.shape[0]
    print(f"Data = {data}, X shape = {X.shape}, n_cells = {n_cells}")
    print(f"ALPHA_LABEL = {ALPHA_LABEL}")

    # 2. Multi-scale UMAP (on PCA embedding)
    scales = generate_scales(
        n_samples=n_cells,
        n_scales=N_SCALES,
        mode=SCALES_MODE,
        min_cap=SCALES_MIN_CAP,
        max_cap=SCALES_MAX_CAP,
    )
    print("scales:", scales)

    X_pca, pca_model = run_pca(X, n_components=PCA_N_COMPONENTS, random_state=PCA_RANDOM_STATE)
    print("X_pca shape:", X_pca.shape)

    embeddings = run_umap_for_scales(
        X_pca,
        scales=scales,
        n_components=UMAP_N_COMPONENTS,
        random_state=1,
        metric=UMAP_METRIC,
        min_dist=UMAP_MIN_DIST,
    )

    for s in scales:
        emb = embeddings[s]
        print(f"scale = {s}, UMAP embedding shape = {emb.shape}")

    # 3. Grassmann distances for each scale (cached)
    os.makedirs(OUT_DIST_DIR, exist_ok=True)
    D_by_scale = {}

    for s, X_umap in embeddings.items():
        cache_path = os.path.join(
            OUT_DIST_DIR,
            f"{data}_scale{s}_umap_chordal.npy"
        )
        D_chordal = get_or_compute_grassmann_distances(
            X_umap,
            metric="chordal",
            cache_path=cache_path,
        )
        print(f"scale={s}, D_chordal shape = {D_chordal.shape}")
        D_by_scale[s] = D_chordal

    # 4. PSL for all cells / scales / k_local
    os.makedirs(OUT_EIGS_DIR, exist_ok=True)

    for s in scales:
        X_umap_s = embeddings[s]
        D_s_raw = D_by_scale[s]
        n_cells_s = X_umap_s.shape[0]

        # cell range for this job (for sbatch parallel)
        center_start = max(0, args.center_start)
        center_end = n_cells_s if args.center_end is None else min(args.center_end, n_cells_s)

        D_s = normalize_distance_matrix(D_s_raw, method=NORMALIZE_METHOD)

        print(f"[info] scale={s}, centers in range [{center_start}, {center_end})")

        for k_local in K_LOCAL_LIST:
            if k_local > n_cells_s:
                print(f"skip: scale={s}, k_local={k_local} > n_cells={n_cells_s}")
                continue

            print(f"[compute] scale={s}, k_local={k_local}, centers[{center_start},{center_end})")

            for center in range(center_start, center_end):
                # Per-cell file name encoding all parameters + cell index
                cell_fname = make_psl_cell_filename(
                    data=DATA_NAME,
                    center=center,
                    scale=s,
                    k_local=k_local,
                    pca=PCA_N_COMPONENTS,
                    umap_dim=UMAP_N_COMPONENTS,
                    umap_metric=UMAP_METRIC,
                    umap_min_dist=UMAP_MIN_DIST,
                    norm_method=NORMALIZE_METHOD,
                    n_filtrations=N_FILTRATIONS,
                    pattern=FILTRATION_PATTERN,
                    dims=PSL_DIMS,
                    rips_max_dim=RIPS_MAX_DIM,
                    sigma=PSL_SIGMA,
                    alpha=ALPHA_LABEL,
                )
                cell_path = os.path.join(OUT_EIGS_DIR, cell_fname)

                # Skip if already computed
                if os.path.exists(cell_path):
                    continue

                # Local patch around this center
                idx, X_local, D_local, center_local_index = get_local_patch(
                    X=X_umap_s,
                    D=D_s,
                    center=center,
                    k=k_local,
                    include_self=True,
                )

                # Max edge length in this patch (under normalized D_local)
                b_max = float(np.max(D_local))
                intervals = generate_filtration_intervals(
                    b_max,
                    n_filtrations=N_FILTRATIONS,
                    pattern=FILTRATION_PATTERN,
                )

                # Compute PSL spectra for each interval separately
                spectra_multi = {int(dim): [] for dim in PSL_DIMS}
                for (a, b) in intervals:
                    spectra = compute_psl_eigs(
                        X_local=X_local,
                        D_local=D_local,
                        a=a,
                        b=b,
                        max_dim=RIPS_MAX_DIM,
                        sigma=PSL_SIGMA,
                        dims=PSL_DIMS,
                        center_index=center_local_index,
                        alpha=ALPHA_LABEL,
                    )
                    for dim in PSL_DIMS:
                        eigs = np.array(spectra[int(dim)], dtype=float)
                        spectra_multi[int(dim)].append(eigs)

                cell_record = {
                    "center": int(center),
                    "idx": idx.astype(int),
                    "intervals": intervals,
                    "spectra": spectra_multi,
                    "meta": {
                        "data_name": DATA_NAME,
                        "scale": int(s),
                        "k_local": int(k_local),
                        "pca_n_components": PCA_N_COMPONENTS,
                        "umap_n_components": UMAP_N_COMPONENTS,
                        "umap_metric": UMAP_METRIC,
                        "umap_min_dist": UMAP_MIN_DIST,
                        "normalize_method": NORMALIZE_METHOD,
                        "n_filtrations": N_FILTRATIONS,
                        "filtration_pattern": FILTRATION_PATTERN,
                        "dims": PSL_DIMS,
                        "rips_max_dim": RIPS_MAX_DIM,
                        "sigma": PSL_SIGMA,
                        "alpha": ALPHA_LABEL,
                    },
                }

                np.save(
                    cell_path,
                    np.array(cell_record, dtype=object),
                    allow_pickle=True,
                )

            print(f"[done] scale={s}, k_local={k_local} for centers[{center_start},{center_end})")


if __name__ == "__main__":
    main()