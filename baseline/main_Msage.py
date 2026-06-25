"""
GraphSAGE (unsupervised) graph-neural-embedding baseline for the HSSE
comparison (reviewer point #1: graph neural embeddings).

Same design as the DGI baseline (main_Mdgi.py): replace the HSSE features by an
*unsupervised* GraphSAGE embedding and keep EVERYTHING else identical
(same preprocessing, same kNN graph, same KFold, same adjust_train_test
oversampling, same StandardScaler, same GBDT classifier, same metrics).

The GraphSAGE encoder is trained with the standard unsupervised objective
(graph-based negative sampling: neighbouring nodes should have similar
embeddings, random nodes dissimilar). Labels are never used; the embedding is
computed once on the full cell graph (transductive) and then evaluated with the
identical 5-fold protocol.
"""

import os
import numpy as np

from auxilary import load_X, load_y

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

# Representation = unsupervised GraphSAGE embedding.
SAGE_HIDDEN = int(os.environ.get("SAGE_HIDDEN", "64"))        # embedding dim
SAGE_KNN = int(os.environ.get("SAGE_KNN", "15"))             # cell-cell kNN graph
SAGE_EPOCHS = int(os.environ.get("SAGE_EPOCHS", "300"))      # unsupervised training
SAGE_LR = float(os.environ.get("SAGE_LR", "0.001"))
PCA_N_COMPONENTS = int(os.environ.get("PCA_N_COMPONENTS", "100"))  # node features
SAGE_RANDOM_STATE = 1

# Classifier: "gbdt" | "logreg" | "linear_svm"  (default gbdt, as in the comparison)
CLASSIFIER = os.environ.get("CLASSIFIER", "gbdt")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_classifier(name):
    """Same classifiers / hyperparameters as the HSSE / PHATE / DGI pipeline."""
    if name == "logreg":
        return LogisticRegression(
            penalty="l2", C=1.0, class_weight=None, max_iter=5000,
            multi_class="multinomial", solver="lbfgs",
        )
    elif name == "linear_svm":
        base = LinearSVC(C=1.0, class_weight=None, max_iter=5000)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    elif name == "gbdt":
        return GradientBoostingClassifier(
            random_state=0, n_estimators=2000, learning_rate=0.002,
            max_depth=7, min_samples_split=5, subsample=0.8, max_features="sqrt",
        )
    else:
        raise ValueError(f"Unknown CLASSIFIER: {name}")


# ======================================================
# GraphSAGE representation (unsupervised)
# ======================================================

class SAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder (standard)."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def compute_sage_features(X, hidden, knn, epochs, lr, pca_n, random_state):
    """
    Build a cell-cell kNN graph on PCA-reduced expression, train an unsupervised
    GraphSAGE encoder on the full graph, and return the embedding as a feature
    matrix of shape (n_cells, hidden). Labels are never used.
    """
    set_seed(random_state)

    n_comp = min(pca_n, X.shape[1], X.shape[0])
    feats = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)

    k = min(knn, X.shape[0] - 1)
    A = kneighbors_graph(feats, n_neighbors=k, mode="connectivity",
                         include_self=False)
    A = A.maximum(A.T)
    coo = A.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float)
    n_nodes = x.size(0)

    model = SAGEEncoder(x.size(1), hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    src, dst = edge_index[0], edge_index[1]
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model(x, edge_index)
        # standard unsupervised GraphSAGE loss: positive = graph edges,
        # negative = random node pairs.
        neg = torch.randint(0, n_nodes, (src.size(0),))
        pos_score = (z[src] * z[dst]).sum(dim=1)
        neg_score = (z[src] * z[neg]).sum(dim=1)
        loss = -(F.logsigmoid(pos_score).mean()
                 + F.logsigmoid(-neg_score).mean())
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"GraphSAGE epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        emb = model(x, edge_index).cpu().numpy()
    return np.asarray(emb, dtype=float)


# ======================================================
# Train/test balancing (identical to HSSE / PHATE / DGI pipeline)
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
    X = load_X(DATA_NAME, DATA_PATH)
    y = load_y(DATA_NAME, DATA_PATH)
    X = np.log10(1 + X).T            # (n_cells, n_genes)

    print("Loaded data:", DATA_NAME)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Representation: GraphSAGE (unsupervised graph neural embedding)")
    print(f"SAGE_HIDDEN={SAGE_HIDDEN}, knn={SAGE_KNN}, epochs={SAGE_EPOCHS}, "
          f"lr={SAGE_LR}, pca={PCA_N_COMPONENTS}")
    print("Classifier:", CLASSIFIER)

    SAGE_features = compute_sage_features(
        X, hidden=SAGE_HIDDEN, knn=SAGE_KNN, epochs=SAGE_EPOCHS,
        lr=SAGE_LR, pca_n=PCA_N_COMPONENTS, random_state=SAGE_RANDOM_STATE,
    )
    print("\nGraphSAGE feature matrix shape:", SAGE_features.shape)
    print("Unique classes:", np.unique(y))
    n_classes = len(np.unique(y))

    n_splits = 5
    icycle = 0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=icycle)

    acc_list, macro_f1_list, weighted_f1_list = [], [], []
    macro_recall_list, mcc_list, auc_macro_list, ba_list = [], [], [], []
    cm_sum = np.zeros((n_classes, n_classes), dtype=float)
    fold_id = 1

    for train_index, test_index in kf.split(SAGE_features):
        print(f"\n=== Fold {fold_id}/{n_splits} ===")
        y_train_full = y[train_index]
        y_test_full = y[test_index]

        y_train_fold, y_test_fold, train_idx_fold, test_idx_fold = adjust_train_test(
            y_train_full, y_test_full, train_index, test_index, random_state=1
        )

        X_train_fold = SAGE_features[train_idx_fold]
        X_test_fold = SAGE_features[test_idx_fold]

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
    print(f"Representation:     GraphSAGE (hidden={SAGE_HIDDEN})")
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
    print(f"SAGE_HIDDEN          = {SAGE_HIDDEN}")
    print(f"SAGE_KNN             = {SAGE_KNN}")
    print(f"SAGE_EPOCHS          = {SAGE_EPOCHS}")
    print(f"SAGE_LR              = {SAGE_LR}")
    print(f"PCA_N_COMPONENTS     = {PCA_N_COMPONENTS}")
    print(f"CLASSIFIER           = {CLASSIFIER}")


if __name__ == "__main__":
    main()
