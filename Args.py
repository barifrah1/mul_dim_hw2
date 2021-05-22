import numpy as np
import pandas as pd
from Consts import *


class Args():

    def __init__(self, *args):
        super(Args, self).__init__(*args)
        self.data = ["Machine learning is super fun",
                     "Python is super, super cool",
                     "Statistics is cool, too",
                     "Data science is fun",
                     "Python is great for machine learning",
                     "I like football",
                     "Football is great to watch"]
