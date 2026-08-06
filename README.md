# HSSE
A Hierarchical Sheaf Spectral Embedding Framework for Single-Cell Analysis
This repository contains the implementation of **HSSE** for single-cell RNA-seq data analysis.

## Environment Setup
We provide a conda environment file: `petlsenv.yml`.

```bash
conda env create -f petlsenv.yml
conda activate petlsenv
```

## Main Pipeline

The workflow consists of three main steps:

### 1. Eigenvalue Computation (Multiscale, Per Cell)

```bash
python main_eigs678.py
```

### 2. Feature Aggregation

```bash
python main_M.py
```
### 3. Feature Extraction and Classification

```bash
python main_Mgbdt.py
```

## Precomputed Eigenvalue Results

Precomputed multiscale persistent sheaf Laplacian eigenvalue results (output of `main_eigs678.py`) for **GSE67835** are provided in `Data/GSE67835/psl_eigs/` to let you skip step 1 above. Each file is a gzip-compressed NumPy array; load directly without a separate decompression step:

```python
import gzip
import numpy as np

with gzip.open("Data/GSE67835/psl_eigs/<filename>.npy.gz", "rb") as f:
    arr = np.load(f)
```

Precomputed results for the other datasets used in this work (GSE45719, GSE75748, GSE82187, GSE84133, GSE94820) are too large for this repository (tens of GB). They will be made available via a separate download link/DOI — see this section for updates.
