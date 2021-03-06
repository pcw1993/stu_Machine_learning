# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/6 6:51 PM
# software: PyCharm

# 逻辑回归预测，判断类别少的
# 分类算法-逻辑回归-二分类，能得出概率值。解决二分类问题利器
# sigmoid函数
# 判断其中一个情况的概率
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def logistic():
    """
    逻辑回归做二分类进行癌症预测
    :return: None
    """
    # 构造标签列名
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses', 'Class']
    # 读取数据
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        names=column_names)
    print(data)

    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)

    data = data.dropna()

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                        test_size=0.25)

    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)

    print(lg.coef_)
    y_predict = lg.predict(x_test)

    print('准确率：', lg.score(x_test, y_test))

    print('召回率：', classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    return None


if __name__ == '__main__':
    logistic()
