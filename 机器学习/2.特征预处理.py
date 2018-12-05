# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/4 4:40 PM
# software: PyCharm

# 通过特定的统计方法（数学方法）将数据转换成算法要求的数据
# preprocessing
# 归一化：特点：通过对原始数据进行变换把数据映射到(默认为[0,1])之间
# 使得某一特征不会对最终结果造成很大影响
# x' = (x-min)/(max-min)  x'' = x`*(mx-mi) + mi
# 标准化


from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np


# sklearn缺失值API:  sklearn.preprocessing.Imputer  要求数据中缺失值为：np.nan
# Imputer(missing_values='NaN', strategy='mean', axis=0)

def mm():
    """
    归一化处理
    :return: None
    """
    # mm = MinMaxScaler(feature_range=(2,3))
    mm = MinMaxScaler()
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)

    return None


def stand():
    """
    标准化缩放
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)

    return None


def im():
    """
    缺失值处理
    :return: None
    """
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 0是列

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)

    return None


def var():
    """
    特征选择--删除低方差特征
    :return: None
    """
    var = VarianceThreshold(threshold=1.0)

    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)

    return None


def pca():
    """
    主成分分析--进行特征降维
    :return: None
    """
    pca = PCA(n_components=0.9)

    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])

    print(data)


    return None


if __name__ == '__main__':
    # mm()
    # stand()
    # im()
    # var()
    pca()
