# Author: Jiang Yikun
# Update date: 2023/4/25
# This utils module is intended to provide the following dicts/functions to the notebook.
# 1. province transformation: 将省份名称转化成GeoJson能识别的省份名称

import numpy as np

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

if __name__ == "__main__":
    # test case
    import pandas as pd
    df = pd.read_csv("./data/rumor_data.csv")
    print(df["province"].apply(lambda x: province_transform(x)))