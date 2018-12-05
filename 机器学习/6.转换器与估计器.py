# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/5 10:57 AM
# software: PyCharm

from sklearn.preprocessing import StandardScaler

s = StandardScaler()

# 转换器
a = s.fit_transform([[1,2,3],[4,5,6]])
print(a)


# 估计器
# estimator
# 1-调用fit， fit(x_train, y_train)

# 2-输入测试集数据
# y_predict =（x_test)
# 预测准确率：score(x_test, y_test)
