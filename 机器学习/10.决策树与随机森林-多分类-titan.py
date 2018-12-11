# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/6 10:28 AM
# software: PyCharm

# 信息熵  H = -(p1logp1 + p2logp2 + ... + p32log32)

# 信息增益  特征A对训练数据集D的信息增益g(D,A),定义为集合D的信息熵H(D)与特征A给定条件下D的信息条件熵H(D|A)之差

# 信息增益高度结果影响越大，作为主要影响标准

# 决策树分类依据之一：信息增益

# ID3
# 信息增益 最大的准则
# C4.5
# 信息增益比 最大的准则
# CART
# 回归树: 平方误差 最小
# 分类树: 基尼系数   最小的准则 在sklearn中可以选择划分的默认原则

# 基尼系数划分更加仔细

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


def decision():
    """
    决策树对泰但尼克号预测生死
    :return: None
    """
    titan = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

    # 处理数据
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    # 处理缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)
    # print(x)

    # 分割训练街，测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理,特征工程，特征，类别，one-hot编码
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    # print(dict.get_feature_names())

    x_test = dict.transform(x_test.to_dict(orient='records'))
    # print(x_train)

    # 用决策树进行预测
    # dec = DecisionTreeClassifier(max_depth=5)
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)

    # 预测准确率
    score = dec.score(x_test, y_test)
    print('预测准确率：', score)

    # 导出决策树结构，graphviz工具dot -Tpng tree.dot -o tree.png 生成p决策树ng
    export_graphviz(dec, out_file='./tree.dot',
                    feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    return None


# 集成学习方法
# 随机森林单个数生成过程：随机在N个样本红选择样本，重复N次，样本可能重复；随机在M个特征中选出m个特征

# 生成多个决策树生成随机森林，样本，特征大多不一样，随机有放回的抽样（bootstrap抽样）
"""
class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’,
 max_depth=None, bootstrap=True, random_state=None)
随机森林分类器
n_estimators：integer，optional（default = 10） 森林里的树木数量
criteria：string，可选（default =“gini”）分割特征的测量方法
max_depth：integer或None，可选（默认=无）树的最大深度 
bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样
max_feature：每个决策树的最大特征数
"""


def trees():
    """
    决策树对泰但尼克号预测生死
    :return: None
    """
    titan = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

    # 处理数据
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    # 处理缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)
    # print(x)

    # 分割训练集，测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理,特征工程，特征，类别，one-hot编码
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    # print(dict.get_feature_names())

    x_test = dict.transform(x_test.to_dict(orient='records'))
    # print(x_train)

    # # 用决策树进行预测
    # # dec = DecisionTreeClassifier(max_depth=5)
    # dec = DecisionTreeClassifier()
    # dec.fit(x_train, y_train)
    #
    # # 预测准确率
    # score = dec.score(x_test, y_test)
    # print('预测准确率：', score)
    #
    # # 导出决策树结构，graphviz工具dot -Tpng tree.dot -o tree.png 生成p决策树ng
    # export_graphviz(dec, out_file='./tree.dot', feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 随机森林进行预测（超参数调优）
    rf = RandomForestClassifier()

    param = {'n_estimators': [100, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}
    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    print('准确率：', gc.score(x_test, y_test))

    print('最优模型参数', gc.best_params_)

    return None


if __name__ == '__main__':
    # decision()
    trees()
