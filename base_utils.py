"""
This utils module is intended to provide basic utilities to the notebook.
"""
from typing import Generator, Any

import numpy as np
import jieba
import jieba.analyse


def province_transform(province: str | float) -> str | float:
    if province is np.NaN:
        return np.NaN

    unique_provinces = {
        "新疆": "新疆维吾尔自治区",
        "西藏": "西藏自治区",
        "内蒙古": "内蒙古自治区",
        "广西": "广西壮族自治区",
        "宁夏": "宁夏回族自治区",
        "香港": "香港特别行政区",
        "澳门": "澳门特别行政区"
    }
    city_provinces = ["重庆", "北京", "天津", "上海"]

    if province in unique_provinces.keys():
        province_trans = unique_provinces[province]
    elif province in city_provinces:
        province_trans = province + "市"
    else:
        province_trans = province + "省"
    return province_trans


def get_chinese_tokens(content: str) -> Generator[str, Any, None]:
    tokens = jieba.cut(content, cut_all=False)
    return tokens


def get_stopwords() -> list[str]:
    stopwords = []
    with open("./data/stop_words", "r", encoding="utf-8") as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    stopwords.append(' ')
    return stopwords


def get_tokens_without_stopwords(tokens: Generator[str, Any, None], stopwords: list[str]) -> list[str]:
    tokens_without_stopwords = []
    for token in tokens:
        if token not in stopwords:
            tokens_without_stopwords.append(token)
    return tokens_without_stopwords


def get_topics(content: str, topK: int = 5) -> list[str]:
    return jieba.analyse.extract_tags(content, topK=topK)
