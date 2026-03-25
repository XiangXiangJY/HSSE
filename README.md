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
