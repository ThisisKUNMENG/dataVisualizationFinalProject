# Author: Jiang Yikun
# Update date: 2023/4/25
# This module is intended to preprocess data so that the notebook is not messy.

from utils import *
import numpy as np
import pandas as pd


def get_preprocessed_data():
    df = pd.read_csv("./data/rumor_data.csv")
    df["province_all"] = df["province"].apply(lambda x: province_transform(x))
    df["tokens"] = df["content"].apply(lambda x: get_chinese_tokens(x))
    df["tokens_without_stopwords"] = df["tokens"].apply(lambda x: get_tokens_without_stopwords(x, get_stopwords()))
    df["topics"] = df["content"].apply(lambda x: get_topics(x))
    return df
