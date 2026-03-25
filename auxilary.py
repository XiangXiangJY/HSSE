# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:46:18 2022

@author: yutah
"""

import numpy as np
import pandas as pd
import csv
import os
from sklearn.cluster import KMeans

def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return

def load_X(data, data_path):
    """
    Load gene expression matrix.
    data_path/data/data_data.csv
    """
    folder = os.path.join(data_path, data)
    file = os.path.join(folder, f"{data}_data.csv")

    if not os.path.exists(file):
        raise FileNotFoundError(f"Cannot find {file}")

    df = pd.read_csv(file)
    X = df.values[:, 1:].astype(float)   # remove first column (cell names)
    return X


def load_y(data, data_path):
    """
    Load labels.
    data_path/data/data_labels.csv
    """
    folder = os.path.join(data_path, data)
    file = os.path.join(folder, f"{data}_labels.csv")

    if not os.path.exists(file):
        raise FileNotFoundError(f"Cannot find {file}")

    df = pd.read_csv(file)
    y = np.array(df["Label"]).astype(int)
    return y



def drop_sample(X, y, min_cell = 15):
    # Used to drop cell types with less than some number
    # X, y: data and label
    # min_cell: minimum number of cells
    original = X.shape[0]
    labels = np.unique(y)
    good_index = []
    for l in labels:
        index = np.where(y == l)[0]
        if index.shape[0] > 15:
            good_index.append(index)
        else:
            print('label %d removed'%(l))
    good_index = np.concatenate(good_index)
    good_index.sort()
    new = good_index.shape[0]
    print(original - new, 'samples removed')
    return X[good_index, :], y[good_index]


def preprocess_data(X, y, min_cell =15):
    # Preprocessing the data, and drop cell types with fewer than min_cell
    X = np.log10(1+X).T
    X, y = drop_sample(X, y, min_cell)
    
    
    return X, y
