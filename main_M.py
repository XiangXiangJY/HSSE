#!/usr/bin/env python
import os
import glob
import numpy as np

# ===================== CONFIG =====================
DATA_NAME = "GSE67835"

# Directory containing per-cell PSL files
PSL_ROOT = "psl_eigs"
PSL_DIR = os.path.join(PSL_ROOT, DATA_NAME)

DIST_TAG = "chordal"

# Normalization method
NORMALIZE_METHOD = "max"
NORM_TAG = f"norm{NORMALIZE_METHOD}"

# The alpha used when computing the restriction maps.
# We will merge only files whose filenames contain this alpha.
ALPHA = 0.01       # <--- change this when evaluating a different alpha

DELETE_ORIGINALS = True
# ===================================================

# Tags that must match the computation settings
PCA_N_COMPONENTS = 50
UMAP_N_COMPONENTS = 20
UMAP_METRIC = "euclidean"
UMAP_MIN_DIST = 0.4
N_FILTRATIONS = 10
FILTRATION_PATTERN = "segments_linear"
PSL_DIMS = (0, 1)
RIPS_MAX_DIM = 2
PSL_SIGMA = None


def format_float_short(x):
    return str(x).replace(".", "p")


def dims_to_str(dims):
    return "d" + "-".join(str(int(d)) for d in dims)


# ===================== FIND CELL FILES =====================
def list_cell_files(psl_dir, data):
    """
    Select per-cell PSL files that match the current run settings,
    including alpha.
    """
    pattern = os.path.join(psl_dir, f"{data}_cell*_psl.npy")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise RuntimeError(f"No PSL files found: {pattern}")

    pca_tag = f"_pca{PCA_N_COMPONENTS}_"
    umap_tag = f"_umap{UMAP_N_COMPONENTS}_"
    dims_tag = f"_{dims_to_str(PSL_DIMS)}_"
    alpha_tag = f"_alpha{format_float_short(ALPHA)}_"

    filtered = []
    for f in all_files:
        base = os.path.basename(f)
        if (
            f"_{NORM_TAG}_" in base and
            pca_tag in base and
            umap_tag in base and
            dims_tag in base and
            alpha_tag in base
        ):
            filtered.append(f)

    if not filtered:
        raise RuntimeError(
            f"No PSL files matched alpha={ALPHA}, norm={NORMALIZE_METHOD}, "
            f"PCA={PCA_N_COMPONENTS}, UMAP={UMAP_N_COMPONENTS}, dims={PSL_DIMS}"
        )

    print(f"[filter] kept {len(filtered)} of {len(all_files)} files for alpha={ALPHA}")
    return filtered


# ===================== PARSE scale, k FROM FILENAME =====================
def parse_scale_k(fname):
    base = os.path.basename(fname)
    parts = base.split("_")

    scale = int([p for p in parts if p.startswith("scale")][0].replace("scale", ""))

    k_candidates = [p for p in parts if p.startswith("k") and not p.startswith("k_local")]
    k_local = int(k_candidates[0].replace("k", ""))

    return scale, k_local


def group_by_scale_k(files):
    groups = {}
    for f in files:
        s, k = parse_scale_k(f)
        groups.setdefault((s, k), []).append(f)
    return groups


# ===================== MERGE ONE GROUP =====================
def merge_group(scale, k_local, file_list, out_dir):

    metric_short = UMAP_METRIC.replace("euclidean", "euc").replace("correlation", "corr")
    md = format_float_short(UMAP_MIN_DIST)
    sigma_str = "auto" if PSL_SIGMA is None else format_float_short(PSL_SIGMA)
    dims_str = dims_to_str(PSL_DIMS)
    alpha_str = format_float_short(ALPHA)

    out_name = (
        f"{DATA_NAME}"
        f"_dist{DIST_TAG}"
        f"_scale{scale}"
        f"_k{k_local}"
        f"_pca{PCA_N_COMPONENTS}"
        f"_umap{UMAP_N_COMPONENTS}"
        f"_met{metric_short}"
        f"_md{md}"
        f"_norm{NORMALIZE_METHOD}"
        f"_f{N_FILTRATIONS}"
        f"_pat{FILTRATION_PATTERN}"
        f"_{dims_str}"
        f"_rips{RIPS_MAX_DIM}"
        f"_sig{sigma_str}"
        f"_alpha{alpha_str}"
        f"_ALLcells_psl.npy"
    )

    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        print(f"[skip] merged file already exists: {out_path}")
        return

    centers, idx_list, spectra_list = [], [], []
    intervals_ref, meta_ref = None, None

    print(f"[merge] scale={scale}, k={k_local}, alpha={ALPHA}, n_files={len(file_list)}")

    for f in sorted(file_list):
        rec = np.load(f, allow_pickle=True).item()

        centers.append(int(rec["center"]))
        idx_list.append(rec["idx"])
        spectra_list.append(rec["spectra"])

        if intervals_ref is None:
            intervals_ref = rec["intervals"]
        else:
            if len(rec["intervals"]) != len(intervals_ref):
                raise RuntimeError(f"Filtration interval mismatch in {f}")

        meta_ref = rec.get("meta", {})

    order = np.argsort(centers)
    centers = np.array(centers)[order]

    merged = {
        "data_name": DATA_NAME,
        "dist_type": DIST_TAG,
        "scale": scale,
        "k_local": k_local,
        "centers": centers,
        "idx_list": [idx_list[i] for i in order],
        "intervals": intervals_ref,
        "spectra_list": [spectra_list[i] for i in order],
        "meta": meta_ref,
    }

    np.save(out_path, np.array(merged, dtype=object), allow_pickle=True)
    print(f"  [saved] {out_path}")

    if DELETE_ORIGINALS:
        for f in file_list:
            try:
                os.remove(f)
                print(f"  deleted {f}")
            except OSError:
                print(f"  failed to delete {f}")


def main():
    print(f"[scan] {PSL_DIR}")
    files = list_cell_files(PSL_DIR, DATA_NAME)
    print(f"[found] {len(files)} per-cell files for alpha={ALPHA}")

    groups = group_by_scale_k(files)
    print(f"[groups] {len(groups)} groups:", sorted(groups.keys()))

    for (s, k), flist in sorted(groups.items()):
        merge_group(s, k, flist, PSL_DIR)


if __name__ == "__main__":
    main()