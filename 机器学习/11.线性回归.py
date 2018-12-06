# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/6 4:49 PM
# software: PyCharm

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def mylinear():
    """
    线性回归预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 标准化处理,目标值要不要进行标准化处理？要
    # 特征值和目标值都要进行标准化处理，要分开进行，实例化两个标准化api
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1,1))
    y_test = std_y.fit_transform(y_test.reshape(-1,1))

    # print(x_train, x_test)

    # estimator预测
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)
    # 预测测试集房价价格
    y_lr_predict = lr.predict(x_test)
    y_lr_predict = std_y.inverse_transform(y_lr_predict)
    # print('每个测试集的预测结果：', y_predict)

    # 梯度下降进行房价预测
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)
    # 预测测试集房价价格
    y_std_predict = sgd.predict(x_test)
    y_std_predict = std_y.inverse_transform(y_std_predict)
    # print('每个测试集的预测结果：', y_predict)

    print('正规方程的均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    print('梯度下降的均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_std_predict))

    return None



if __name__ == '__main__':
    mylinear()
