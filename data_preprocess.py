"""
This module is intended to preprocess data so that the notebook is not messy.
"""

from utils import *
import numpy as np
import pandas as pd


def get_preprocessed_data():
    df = pd.read_csv("./data/rumor_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df["log_like"] = df["like"].apply(lambda x: np.log(x + 1))
    df["province_all"] = df["province"].apply(lambda x: province_transform(x))
    df["tokens"] = df["content"].apply(lambda x: get_chinese_tokens(x))
    df["tokens_without_stopwords"] = df["tokens"].apply(lambda x: get_tokens_without_stopwords(x, get_stopwords()))
    df["topics"] = df["content"].apply(lambda x: get_topics(x))
    return df
