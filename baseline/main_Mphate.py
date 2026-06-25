"""
PHATE-representation baseline for the HSSE comparison (reviewer point #1).

Idea: replace the HSSE persistent-spectral features by a PHATE embedding,
keep EVERYTHING else identical to the HSSE classification pipeline
(same preprocessing, same KFold, same adjust_train_test oversampling,
same StandardScaler, same classifiers LR / Linear SVM / GBDT).

This yields a clean "fixed-classifier, swap-representation" comparison.
"""

import os
import numpy as np

from auxilary import load_X, load_y

import phate

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
# Config
# ======================================================

DATA_NAME = os.environ.get("DATA_NAME", "GSE67835")
DATA_PATH = "./data/"

# Representation = PHATE embedding.
# PHATE defaults to 2 components (for visualization); for a fair
# discriminative baseline we allow more components.
PHATE_N_COMPONENTS = int(os.environ.get("PHATE_N_COMPONENTS", "20"))
PHATE_KNN = int(os.environ.get("PHATE_KNN", "5"))      # PHATE default
PHATE_DECAY = int(os.environ.get("PHATE_DECAY", "40")) # PHATE default
PHATE_RANDOM_STATE = 1

# Classifier: "logreg" | "linear_svm" | "gbdt"
CLASSIFIER = os.environ.get("CLASSIFIER", "logreg")


def get_classifier(name):
    """Same classifiers / hyperparameters as the HSSE pipeline."""
    if name == "logreg":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight=None,
            max_iter=5000,
            multi_class="multinomial",
            solver="lbfgs",
        )
    elif name == "linear_svm":
        base = LinearSVC(C=1.0, class_weight=None, max_iter=5000)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    elif name == "gbdt":
        return GradientBoostingClassifier(
            random_state=0,
            n_estimators=2000,
            learning_rate=0.002,
            max_depth=7,
            min_samples_split=5,
            subsample=0.8,
            max_features="sqrt",
        )
    else:
        raise ValueError(f"Unknown CLASSIFIER: {name}")


# ======================================================
# PHATE representation
# ======================================================

def compute_phate_features(X, n_components, knn, decay, random_state):
    """
    Fit PHATE once on the full (n_cells x n_genes) matrix and return the
    embedding as a feature matrix of shape (n_cells, n_components).
    """
    op = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        random_state=random_state,
        verbose=1,
    )
    emb = op.fit_transform(X)
    return np.asarray(emb, dtype=float)


# ======================================================
# Train/test balancing (identical to HSSE pipeline)
# ======================================================

def adjust_train_test(y_train, y_test, train_index, test_index, random_state=1):
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
    # 1. Load and preprocess data EXACTLY as in the HSSE pipeline
    X = load_X(DATA_NAME, DATA_PATH)
    y = load_y(DATA_NAME, DATA_PATH)
    X = np.log10(1 + X).T            # (n_cells, n_genes)

    n_cells = X.shape[0]
    print("Loaded data:", DATA_NAME)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Representation: PHATE")
    print(f"PHATE_N_COMPONENTS={PHATE_N_COMPONENTS}, knn={PHATE_KNN}, decay={PHATE_DECAY}")
    print("Classifier:", CLASSIFIER)

    # 2. PHATE representation (fit once on full data; transductive, same as HSSE)
    PHATE_features = compute_phate_features(
        X,
        n_components=PHATE_N_COMPONENTS,
        knn=PHATE_KNN,
        decay=PHATE_DECAY,
        random_state=PHATE_RANDOM_STATE,
    )
    print("\nPHATE feature matrix shape:", PHATE_features.shape)
    print("Unique classes:", np.unique(y))
    n_classes = len(np.unique(y))

    # 3. 5-fold KFold + adjust_train_test + classifier (identical protocol)
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

    for train_index, test_index in kf.split(PHATE_features):
        print(f"\n=== Fold {fold_id}/{n_splits} ===")

        y_train_full = y[train_index]
        y_test_full = y[test_index]

        y_train_fold, y_test_fold, train_idx_fold, test_idx_fold = adjust_train_test(
            y_train_full, y_test_full, train_index, test_index, random_state=1
        )

        X_train_fold = PHATE_features[train_idx_fold]
        X_test_fold = PHATE_features[test_idx_fold]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        clf = get_classifier(CLASSIFIER)
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
                y_test_fold, y_proba, multi_class="ovr", average="macro"
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
    print(f"Representation:     PHATE (n_components={PHATE_N_COMPONENTS})")
    print(f"Classifier:         {CLASSIFIER}")
    print(f"Balanced Accuracy:  {ba_mean:.4f} ± {ba_std:.4f}")
    print(f"Accuracy:           {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Macro F1:           {macro_f1_mean:.4f} ± {macro_f1_std:.4f}")
    print(f"Weighted F1:        {weighted_f1_mean:.4f} ± {weighted_f1_std:.4f}")
    print(f"Macro recall:       {macro_recall_mean:.4f} ± {macro_recall_std:.4f}")
    print(f"MCC:                {mcc_mean:.4f} ± {mcc_std:.4f}")
    print(f"Macro-AUC (OVR):    {auc_mean:.4f} ± {auc_std:.4f}")
    print("\nAverage confusion matrix over folds (rows=true, cols=pred):")
    print(avg_cm)

    print("\n===== Parameter Summary =====")
    print(f"DATA_NAME            = {DATA_NAME}")
    print(f"PHATE_N_COMPONENTS   = {PHATE_N_COMPONENTS}")
    print(f"PHATE_KNN            = {PHATE_KNN}")
    print(f"PHATE_DECAY          = {PHATE_DECAY}")
    print(f"CLASSIFIER           = {CLASSIFIER}")


if __name__ == "__main__":
    main()
