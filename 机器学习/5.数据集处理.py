# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/5 10:36 AM
# software: PyCharm

from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split

# 加载并返回鸢尾花数据集
li = load_iris()

# print('特征值',li.data)
# print('目标值',li.target)
# print(li.DESCR)

# 注意返回值，包含训练集train  x_train, y_train     测试集test x_trest, y_test
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
# print('训练集', x_train, y_train)
#
# print('测试集',x_test, y_test)


news = fetch_20newsgroups(subset='all')
print(news.data)
print(news.target)


lb = load_boston()
print(lb.data)
print(lb.target)
print(lb.DESCR)





