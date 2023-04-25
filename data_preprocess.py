# Author: Jiang Yikun
# Update date: 2023/4/25
# This module is intended to preprocess data so that the notebook is not messy.

from utils import province_transform
import numpy as np
import pandas as pd

def get_preprocessed_data():
    df = pd.read_csv("./data/rumor_data.csv")
    df["province_all"] = df["province"].apply(lambda x: province_transform(x))
    return df