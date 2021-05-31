import numpy as np
import pandas as pd
from Args import Args
from Dataset import Dataset
from LatentSemanticIndexing import LatentSemanticIndexing, plotDocsAndTermsGraph, word_doc_query


if __name__ == "__main__":
    args = Args()
    dataset = Dataset(args.data)
    lsi = LatentSemanticIndexing(dataset.df, 2)
    composedForm = lsi.composedForm
    print(composedForm)
    docsVectorsComposed, termsVectorsComposed = lsi.getTermsAndDocsVectors()
    print("docs vectors: ", docsVectorsComposed)
    print("terms vectors: ", termsVectorsComposed)
    plotDocsAndTermsGraph(docsVectorsComposed,termsVectorsComposed, dataset.docs, dataset.terms)

    word_component_df = pd.DataFrame(termsVectorsComposed.T, index=["component_1", "component_2"], columns = lsi.df.columns)
    word1 = input("Enter 1st word:")
    word2 = input("Enter 2nd word:")
    #word1 = 'cool'
    #word2 = 'super'

    word_doc_query(word1, word2, word_component_df,docsVectorsComposed,termsVectorsComposed, dataset.docs, dataset.terms)
