# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/7 10:47 AM
# software: PyCharm

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# with open('./data/1.txt', 'r') as f:
#     cont = f.read()
# print(cont)

def kmeans():
    # 合并表
    prior = pd.read_csv('./data/order_products__prior.csv')
    products = pd.read_csv('./data/products.csv')
    orders = pd.read_csv('./data/orders.csv')
    aisles = pd.read_csv('./data/aisles.csv')
    _mg = pd.merge(prior, products, on=['product_id', 'product_id'])
    _mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
    mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])

    print(mt.head(10))

    # 交叉表（特殊分组工具）
    cross = pd.crosstab(mt['user_id'], mt['aisle'])
    print(cross.head(10))

    # 进行降维，主成本分析
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(cross)
    print(data)

    print(data.shape)  # 降维



    # 聚类
    # 减少样本数量
    x = data[:500]

    km = KMeans(n_clusters=4)
    km.fit(x)
    predict = km.predict(x)
    print(predict)

    # 显示聚类结果
    plt.figure(figsize=(10,10))

    # 建立四个颜色的列表
    colored = ['orange', 'green', 'blue', 'purple']
    colr = [colored[i] for i in predict]
    plt.scatter(x[:,1], x[:,20], color=colr)
    plt.xlabel('1')
    plt.ylabel('20')
    plt.show()

    # 评判聚类效果，轮廓系数
    score = silhouette_score(x, predict)
    print(score)




if __name__ == '__main__':
    kmeans()

