import numpy as np
import pandas as pd
import random as random
from sklearn.feature_extraction.text import CountVectorizer
#from Consts import NUM_OF_SUBSETS


class Dataset():
    def __init__(self, *args, **keywords):
        self.data = args[0]
        self.docs = list(range(len(self.data)))
        vectorizer = CountVectorizer(min_df=1, stop_words='english')
        self.data = vectorizer.fit_transform(self.data)
        self.terms = vectorizer.get_feature_names()
        self.df = pd.DataFrame(self.data.toarray(), columns=self.terms
                               )
        print(self.df)
