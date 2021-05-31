from os import error
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
import scipy
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


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


def plotDocsAndTermsGraph(docsVecs, termVecs, docsNames, termsNames, query_loc='NA',closest_doc_loc = 'NA', word1_loc = 'NA', word2_loc = 'NA'):

    plt.scatter(docsVecs[:, 0], docsVecs[:, 1], color='blue')
    # docs plotting
    for i in range(len(docsVecs)):

        label = f"d{docsNames[i]}"

        plt.annotate(label,  # this is the text
                     # this is the point to label
                     (docsVecs[i][0], docsVecs[i][1]),
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')
    plt.scatter(termVecs[:, 0], termVecs[:, 1], color='green')
    for j in range(len(termVecs)):
        label = f"{termsNames[j]}"

        plt.annotate(label,  # this is the text
                     # this is the point to label
                     (termVecs[j][0], termVecs[j][1]),
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')
    if query_loc is not 'NA':
        label = 'query'
        plt.annotate("query",query_loc, textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')
        plt.scatter(query_loc[0], query_loc[1], color='red')

        plt.annotate("", closest_doc_loc, textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')
        plt.scatter(closest_doc_loc[0], closest_doc_loc[1], s = 250, facecolors='none', color='red',marker='o')
        plt.scatter(word1_loc[0], word1_loc[1], s=250, facecolors='none', color='red', marker='o')
        plt.scatter(word2_loc[0], word2_loc[1], s=250, facecolors='none', color='red', marker='o')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Geometrical plot")

    plt.show()

def word_doc_query(word1, word2, word_component_df,docsVecs, termVecs, docsNames, termsNames):
    word1_loc = word_component_df[word1]
    word2_loc = word_component_df[word2]
    query_loc = [(word1_loc[0] + word2_loc[0]) / 2, (word1_loc[1] + word2_loc[1]) / 2]
    """
        idx = (np.linalg.norm(docsVecs - query_loc)).argmin()
        d=[]
        for i in range docsVecs:
            d=cdist(docsVecs[i], query_loc, 'euclidean')
    """

    kdtree = KDTree(docsVecs)
    d, closest_doc = kdtree.query((query_loc))
    print("closest document location:", docsVecs[closest_doc],"\nclosest document number:", closest_doc)
    closest_doc_loc = docsVecs[closest_doc]


    plotDocsAndTermsGraph(docsVecs, termVecs, docsNames, termsNames,query_loc, closest_doc_loc, word1_loc, word2_loc)







