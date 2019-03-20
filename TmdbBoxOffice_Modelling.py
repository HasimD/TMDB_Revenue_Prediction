# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold


train = pd.read_csv("train_features.csv") 
test = pd.read_csv("test_features.csv")

