import numpy as np
import pandas as pd
from Args import Args
from Dataset import Dataset
from LatentSemanticIndexing import LatentSemanticIndexing, plotDocsAndTermsGraph


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
