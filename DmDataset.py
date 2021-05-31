import numpy as np
import pandas as pd
import random as random
from sklearn.feature_extraction.text import CountVectorizer


class DmDataset():
    def __init__(self, *args, **keywords):
        self.data = pd.read_csv(keywords["path"], names=['x', 'y']).values
