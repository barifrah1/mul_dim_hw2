from os import error
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
import scipy
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform, pdist


class DiffusionMaps():
    def __init__(self, dataset: np.array, C: float, d: int):
        self.dataset = dataset
        self.C = C
        self.d = d


def myDM(dataset, C, d):
    NUMBER_OF_ROWS = len(dataset)
    NUMBER_OF_COLS = len(dataset[0])
    dist = squareform(pdist(dataset))
    print(dist)
    minDistInRow = 100000000*np.ones(NUMBER_OF_ROWS)
    for j in range(NUMBER_OF_ROWS):
        for i in range(NUMBER_OF_ROWS):
            if(i != j and dist[j, i] < minDistInRow[j]):
                minDistInRow[j] = dist[j, i]
    epsilon = max(minDistInRow)
    updatedDist = -(dist**2)/(C*epsilon)
    W = np.exp(updatedDist)
    D = np.zeros(NUMBER_OF_ROWS)
    for idx, row in enumerate(W):
        D[idx] = sum(row)
    InvertedD = D**-1
    P = W
    for i, row in enumerate(P):
        P[i] = P[i]*InvertedD[i]
    U, S, Vt = svds(P, k=d)
    sorted_index = np.argsort(S)[::-1]
    S = S[sorted_index]
    U = U[:, sorted_index]
    Vt = Vt[sorted_index, :]
    dm_values = np.zeros((NUMBER_OF_ROWS, d))
    for j in range(1, d):
        dm_values[:, j-1] = S[j]*U[:, j]
    fiedlerVector = S[-2]*U[:, -2]
    return dm_values, fiedlerVector
