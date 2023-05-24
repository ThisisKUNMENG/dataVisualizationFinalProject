"""
This module is intended to preprocess data so that the notebook is not messy.
"""

from base_utils import *
import numpy as np
import pandas as pd


def get_preprocessed_data():
    df = pd.read_csv('./data/rumor_data_new.csv', index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df['log_like'] = np.log(df['like'] + 1)
    df['province'] = df['province'].apply(lambda x: province_transform(x))  # 将省份名称转化成GeoJson能识别的省份名称
    # df["province_all"] = df["province"].apply(lambda x: province_transform(x))
    df['content_token'] = df['content'].apply(
        lambda x: get_tokens_without_stopwords(get_chinese_tokens(x), stopwords=get_stopwords())
    )
    return df
