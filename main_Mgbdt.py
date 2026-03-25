

import os
import glob
import numpy as np

from auxilary import load_X, load_y

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)

# ======================================================
# Global parameters – must match your PSL pipeline
# ======================================================

DATA_NAME = "GSE67835"
DATA_PATH = "./data/"
PSL_ROOT = "./psl_eigs"
PSL_DIR = os.path.join(PSL_ROOT, DATA_NAME)

DIST_TAG = "chordal"

PCA_N_COMPONENTS = 50
UMAP_N_COMPONENTS = 20
UMAP_METRIC = "euclidean"
UMAP_MIN_DIST = 0.4

NORMALIZE_METHOD = "max"
N_FILTRATIONS = 10
FILTRATION_PATTERN = "segments_linear"

PSL_DIMS = (0,1)
RIPS_MAX_DIM = 2
PSL_SIGMA = None

# alpha used in restriction map and in merged filenames
ALPHA = 0.01  # must match the alpha used in PSL generation and merge

# Which PSL dims to use as features
DIMS_USE = (0,1)

# summarize_eigs   [sum, mean, max, min, std]
STATS_PER_DIM = {
    0: [0, 1, 2, 3, 4],
   # 1: [0, 1, 2, 3, 4],
}

# (scale, k) combos to include in features
SCALES = [5, 23, 46, 72, 100]#[5, 14, 25, 37, 50] #
K_LIST = [5, 10, 15, 20, 30, 40,50,60,70,80]


def format_float_short(x):
    return str(x).replace(".", "p")


# ======================================================
# Helpers for PSL features
# ======================================================

def summarize_eigs(eigs, eps=1e-8):
    """
    Given a 1D array of eigenvalues, return
      [sum_pos, mean_pos, max_pos, min_pos, std_pos]
    over positive eigenvalues (eigs > eps).
    If there are no positive eigenvalues, return zeros.
    """
    eigs = np.asarray(eigs, dtype=float)
    pos = eigs[eigs > eps]

    if pos.size == 0:
        return np.zeros(5, dtype=float)

    s = float(pos.sum())
    m = float(pos.mean())
    mx = float(pos.max())
    mn = float(pos.min())
    sd = float(pos.std(ddof=0))

    return np.array([s, m, mx, mn, sd], dtype=float)


def load_psl_features_block(
    psl_dir,
    data_name,
    n_cells,
    scale,
    k_local,
    dims_use=DIMS_USE,
):
    """
    Load PSL spectra from a merged *_ALLcells_psl.npy file and convert them
    into a feature matrix of shape (n_cells, feat_dim).

    We look for merged files of the form:

      {data_name}_dist{DIST_TAG}_scale{scale}_k{k_local}_..._alpha{ALPHA}_ALLcells_psl.npy

    and then:
      - For each cell's spectra, apply summarize_eigs on each interval and dim.
      - Assemble feats with shape (n_cells, feat_dim).
    """
    alpha_str = format_float_short(ALPHA)

    pattern = os.path.join(
        psl_dir,
        f"{data_name}_dist{DIST_TAG}_scale{scale}_k{k_local}_*"
        f"_alpha{alpha_str}_ALLcells_psl.npy"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No merged PSL file found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[warning] Multiple merged files found, using first: {matches}")
    path = matches[0]
    print(f"[load merged] {path}")

    merged = np.load(path, allow_pickle=True).item()

    centers = np.asarray(merged["centers"], dtype=int)
    spectra_list = merged["spectra_list"]
    intervals = merged["intervals"]

    if centers.shape[0] != len(spectra_list):
        raise ValueError(
            f"centers length {centers.shape[0]} != spectra_list length {len(spectra_list)}"
        )

    n_intervals = len(intervals)

    stats_count_per_dim = []
    for d in dims_use:
        dim_key = int(d)
        idxs = STATS_PER_DIM.get(dim_key, [0, 1, 2, 3, 4])
        stats_count_per_dim.append(len(idxs))
    total_stats_per_interval = sum(stats_count_per_dim)

    feat_dim = total_stats_per_interval * n_intervals
    feats = np.zeros((n_cells, feat_dim), dtype=float)

    for i, center in enumerate(centers):
        spectra = spectra_list[i]  # dict: dim -> list over intervals

        row_parts = []
        for d in dims_use:
            dim_key = int(d)
            eigs_list = spectra.get(dim_key, [])

            stat_indices = STATS_PER_DIM.get(dim_key, [0, 1, 2, 3, 4])

            if len(eigs_list) == 0:
                for _ in range(n_intervals):
                    row_parts.append(np.zeros(len(stat_indices), dtype=float))
            else:
                if len(eigs_list) != n_intervals:
                    raise ValueError(
                        f"spectra[dim={dim_key}] has {len(eigs_list)} intervals "
                        f"but global intervals has {n_intervals} "
                        f"(center={center}, scale={scale}, k={k_local})"
                    )
                for eigs in eigs_list:
                    full_stats = summarize_eigs(eigs)
                    row_parts.append(full_stats[stat_indices])

        row_vec = np.concatenate(row_parts)
        if center < 0 or center >= n_cells:
            raise ValueError(f"center index {center} out of range [0, {n_cells})")
        feats[center, :] = row_vec

    return feats


# ======================================================
# GBDT part
# ======================================================

def adjust_train_test(y_train, y_test, train_index, test_index, random_state=1):
    """
    Adjust train/test sets:

    - Keep only classes that appear in both train and test
    - Require: train count > 5, test count > 3 for each class
    - Oversample each class in the training set to 5 * avgCount
    """
    rng = np.random.default_rng(random_state)

    unique_labels_temp = np.intersect1d(y_train, y_test)
    unique_labels_temp.sort()

    unique_labels = []
    counter = []
    new_test_index_list = []

    for l in unique_labels_temp:
        l_train = np.where(l == y_train)[0]
        l_test = np.where(l == y_test)[0]
        if l_train.shape[0] > 5 and l_test.shape[0] > 3:
            unique_labels.append(l)
            new_test_index_list.append(l_test)
            counter.append(l_train.shape[0])

    if len(unique_labels) == 0:
        return y_train, y_test, train_index, test_index

    new_test_index_local = np.concatenate(new_test_index_list)
    new_test_index_local.sort()
    new_y_test = y_test[new_test_index_local]
    new_test_index = test_index[new_test_index_local]

    new_train_index_list = []
    avgCount = int(np.ceil(np.mean(counter)))
    for l in unique_labels:
        l_train = np.where(l == y_train)[0]
        index = rng.choice(l_train, size=5 * avgCount, replace=True)
        new_train_index_list.append(index)

    new_train_index_local = np.concatenate(new_train_index_list)
    new_train_index_local.sort()
    new_y_train = y_train[new_train_index_local]
    new_train_index = train_index[new_train_index_local]

    return new_y_train, new_y_test, new_train_index, new_test_index


def main():
    # 1. Load raw data and labels
    X_raw = load_X(DATA_NAME, DATA_PATH)
    y = load_y(DATA_NAME, DATA_PATH)
    X_raw = X_raw.T

    n_cells = X_raw.shape[0]
    print("Loaded data:", DATA_NAME)
    print("X_raw shape:", X_raw.shape, "y shape:", y.shape)

    # 2. Aggregate PSL features across (scale, k)
    feat_blocks = []

    for s in SCALES:
        for k_local in K_LIST:
            print(f"\nLoading PSL features for scale={s}, k={k_local}, alpha={ALPHA} ...")
            feats_sk = load_psl_features_block(
                psl_dir=PSL_DIR,
                data_name=DATA_NAME,
                n_cells=n_cells,
                scale=s,
                k_local=k_local,
                dims_use=DIMS_USE,
            )
            print(
                f"Block shape (scale={s}, k={k_local}): "
                f"{feats_sk.shape}, dims_use={DIMS_USE}"
            )
            feat_blocks.append(feats_sk)

    PSL_features = np.concatenate(feat_blocks, axis=1)
    print("\nFinal PSL feature matrix shape:", PSL_features.shape)
    print("Unique classes:", np.unique(y))
    n_classes = len(np.unique(y))

    # 3. 5-fold KFold + adjust_train_test + GBDT
    n_splits = 5
    icycle = 0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=icycle)

    acc_list = []
    macro_f1_list = []
    weighted_f1_list = []
    macro_recall_list = []
    mcc_list = []
    auc_macro_list = []
    ba_list = []

    cm_sum = np.zeros((n_classes, n_classes), dtype=float)

    fold_id = 1

    for train_index, test_index in kf.split(PSL_features):
        print(f"\n=== Fold {fold_id}/{n_splits} ===")

        y_train_full = y[train_index]
        y_test_full = y[test_index]

        y_train_fold, y_test_fold, train_idx_fold, test_idx_fold = adjust_train_test(
            y_train_full, y_test_full, train_index, test_index, random_state=1
        )

        X_train_fold = PSL_features[train_idx_fold]
        X_test_fold = PSL_features[test_idx_fold]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        clf = GradientBoostingClassifier(
            random_state=0,
            n_estimators=2000,
            learning_rate=0.002,
            max_depth=7,
            min_samples_split=5,
            subsample=0.8,
            max_features="sqrt",
        )
        clf.fit(X_train_scaled, y_train_fold)

        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test_fold, y_pred)
        macro_f1 = f1_score(y_test_fold, y_pred, average="macro")
        weighted_f1 = f1_score(y_test_fold, y_pred, average="weighted")
        macro_recall = recall_score(y_test_fold, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test_fold, y_pred)
        ba = balanced_accuracy_score(y_test_fold, y_pred)

        try:
            auc_macro = roc_auc_score(
                y_test_fold,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except ValueError as e:
            print(f"Warning: ROC AUC calculation failed for fold {fold_id}: {e}")
            auc_macro = np.nan

        cm = confusion_matrix(y_test_fold, y_pred, labels=np.unique(y))
        cm_sum += cm

        print(f"Fold {fold_id} balanced acc:     {ba:.4f}")
        print(f"Fold {fold_id} accuracy:         {acc:.4f}")
        print(f"Fold {fold_id} macro F1:         {macro_f1:.4f}")
        print(f"Fold {fold_id} weighted F1:      {weighted_f1:.4f}")
        print(f"Fold {fold_id} macro recall:     {macro_recall:.4f}")
        print(f"Fold {fold_id} MCC:              {mcc:.4f}")
        print(f"Fold {fold_id} Macro-AUC (OVR):  {auc_macro:.4f}")
        print("Fold classification report:")
        print(classification_report(y_test_fold, y_pred))

        acc_list.append(acc)
        macro_f1_list.append(macro_f1)
        weighted_f1_list.append(weighted_f1)
        macro_recall_list.append(macro_recall)
        mcc_list.append(mcc)
        auc_macro_list.append(auc_macro)
        ba_list.append(ba)

        fold_id += 1

    def mean_std(x):
        x = np.asarray(x, dtype=float)
        return x.mean(), x.std(ddof=0)

    ba_mean, ba_std = mean_std(ba_list)
    acc_mean, acc_std = mean_std(acc_list)
    macro_f1_mean, macro_f1_std = mean_std(macro_f1_list)
    weighted_f1_mean, weighted_f1_std = mean_std(weighted_f1_list)
    macro_recall_mean, macro_recall_std = mean_std(macro_recall_list)
    mcc_mean, mcc_std = mean_std(mcc_list)

    auc_clean = [a for a in auc_macro_list if not np.isnan(a)]
    if len(auc_clean) > 0:
        auc_mean, auc_std = mean_std(auc_clean)
    else:
        auc_mean, auc_std = np.nan, np.nan

    avg_cm = cm_sum / n_splits

    print("\n===== 5-fold Cross-Validation Summary =====")
    print(f"Balanced Accuracy:  {ba_mean:.4f} ± {ba_std:.4f}")
    print(f"Accuracy:           {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Macro F1:           {macro_f1_mean:.4f} ± {macro_f1_std:.4f}")
    print(f"Weighted F1:        {weighted_f1_mean:.4f} ± {weighted_f1_std:.4f}")
    print(f"Macro recall:       {macro_recall_mean:.4f} ± {macro_recall_std:.4f}")
    print(f"MCC:                {mcc_mean:.4f} ± {mcc_std:.4f}")
    print(f"Macro-AUC (OVR):    {auc_mean:.4f} ± {auc_std:.4f}")
    print("\nAverage confusion matrix over folds (rows=true, cols=pred):")
    print(avg_cm)

    print("\n===== Parameter Summary for This Run =====")
    print("\n--- PSL / Geometry Parameters ---")
    print(f"DATA_NAME            = {DATA_NAME}")
    print(f"PCA_N_COMPONENTS     = {PCA_N_COMPONENTS}")
    print(f"UMAP_N_COMPONENTS    = {UMAP_N_COMPONENTS}")
    print(f"UMAP_METRIC          = {UMAP_METRIC}")
    print(f"UMAP_MIN_DIST        = {UMAP_MIN_DIST}")
    print(f"NORMALIZE_METHOD     = {NORMALIZE_METHOD}")
    print(f"N_FILTRATIONS        = {N_FILTRATIONS}")
    print(f"FILTRATION_PATTERN   = {FILTRATION_PATTERN}")
    print(f"PSL_DIMS             = {PSL_DIMS}")
    print(f"RIPS_MAX_DIM         = {RIPS_MAX_DIM}")
    print(f"PSL_SIGMA            = {PSL_SIGMA}")
    print(f"SCALES (list)        = {SCALES}")
    print(f"K_LIST               = {K_LIST}")
    print(f"DIMS_USE             = {DIMS_USE}")
    print(f"STATS_PER_DIM        = {STATS_PER_DIM}")
    print(f"DIST_TAG             = {DIST_TAG}")
    print(f"ALPHA                = {ALPHA}")

    print("\n--- Gradient Boosting (GBDT) Parameters ---")
    clf = GradientBoostingClassifier(
        random_state=0,
        n_estimators=2000,
        learning_rate=0.002,
        max_depth=7,
        min_samples_split=5,
        subsample=0.8,
        max_features="sqrt",
    )
    for k, v in clf.get_params().items():
        print(f"{k:20s} = {v}")


if __name__ == "__main__":
    main()