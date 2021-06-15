import numpy as np
import pandas as pd
from Args import Args
from Dataset import Dataset
from DmDataset import DmDataset
from LatentSemanticIndexing import LatentSemanticIndexing, plotDocsAndTermsGraph, word_doc_query
from DiffusionMaps import DiffusionMaps, myDM
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Consts import *

if __name__ == "__main__":
    args = Args()
    dataset = Dataset(args.data)
    lsi = LatentSemanticIndexing(dataset.df, 2)
    composedForm = lsi.composedForm
    print(composedForm)
    docsVectorsComposed, termsVectorsComposed = lsi.getTermsAndDocsVectors()
    print("docs vectors: ", docsVectorsComposed)
    print("terms vectors: ", termsVectorsComposed)
    plotDocsAndTermsGraph(docsVectorsComposed,
                          termsVectorsComposed, dataset.docs, dataset.terms)

    word_component_df = pd.DataFrame(termsVectorsComposed.T, index=[
                                     "component_1", "component_2"], columns=lsi.df.columns)
    word1 = input("Enter 1st word:")
    word2 = input("Enter 2nd word:")
    # word1 = 'cool'
    # word2 = 'super'

    word_doc_query(word1, word2, word_component_df, docsVectorsComposed,
                   termsVectorsComposed, dataset.docs, dataset.terms)

"""
    # q3
    dataset = DmDataset(path=DataPathQ3)
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("data plotting")
    plt.show()

    dm = DiffusionMaps(dataset.data, C, NUMBER_OF_DIMENSIONS_Q3)
    dmValues, fiedlerVector = myDM(dm.dataset, dm.C, dm.d)

    # plot 3 dm
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, -60)
    scatter = ax.scatter(dmValues[:, 0], dmValues[:, 1], dmValues[:, 2],
                         alpha=0.6)
    # chart
    plt.title("3 dm coordinates")
    ax.set_xlabel('dm1')
    ax.set_ylabel('dm2')
    ax.set_zlabel('dm3')
    plt.show()
    # k-means
    model = KMeans(n_clusters=K_MEANS_K)
    model.fit(dataset.data)
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=model.labels_)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Clustering original data")
    plt.show()

    # plot according to fiedler vector
    #fiedler = dmValues[:, -2]
    fiedlerClustering = np.fromiter(
        map(lambda x: 1 if x >= 0 else 0, fiedlerVector), dtype=np.int)
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=fiedlerClustering)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Clustering by fiedler")
    plt.show()
"""
