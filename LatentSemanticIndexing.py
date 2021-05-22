from os import error
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
import scipy
from scipy.sparse.linalg import svds


class LatentSemanticIndexing():
    def __init__(self, df: DataFrame, k: int):
        self.df = df
        self.k = k
        # frequency matrix num of termx X num of docs
        self.A = scipy.sparse.coo_matrix.asfptype(df.values)
        self.U, self.Sigma, self.V_t = svds(self.A, k=self.k)
        self.SigmaI = self.Sigma*np.eye(self.k)
        self.composedForm = self.getComposedForm()

    def getComposedForm(self):
        try:
            k = self.k
            return self.U[:, 0:k]@self.SigmaI[0:k, 0:k]@self.V_t[0:k, :]
        except error:
            raise(error)

    def getTermsAndDocsVectors(self):
        k = self.k
        docsVectorsComposed = self.U[:, 0:k]@self.SigmaI[0:k, 0:k]
        termsVectorsComposed = self.SigmaI[0:k,
                                           0:k]@self.V_t[0:k, :]
        return docsVectorsComposed, termsVectorsComposed.transpose()


def plotDocsAndTermsGraph(docsVecs, termVecs, docsNames, termsNames):

    plt.scatter(docsVecs[:, 0], docsVecs[:, 1], color='red')
    # docs plotting
    for i in range(len(docsVecs)):

        label = f"d{docsNames[i]}"

        plt.annotate(label,  # this is the text
                     # this is the point to label
                     (docsVecs[i][0], docsVecs[i][1]),
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')
    plt.scatter(termVecs[:, 0], termVecs[:, 1], color='blue')
    for j in range(len(termVecs)):
        label = f"{termsNames[j]}"

        plt.annotate(label,  # this is the text
                     # this is the point to label
                     (termVecs[j][0], termVecs[j][1]),
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Geometrical plot")

    plt.show()
