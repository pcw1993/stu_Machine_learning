# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/4 7:11 PM
# software: PyCharm

# https://www.kaggle.com/c/instacart-market-basket-analysis/data
# products.csv               商品信息
# order_products__prior.csv  订单与商品信息
# orders.csv                 用户的订单信息
# aisles.csv                 商品所属具体物品类别

# 合并各表，jupyter notebook
# order_product: order_id,product_id
# products:product_id,aisle_id
# orders:order_id,user_id
# aisles:aisle_id,aisle

import pandas as pd
from sklearn.decomposition import PCA

# with open('./data/1.txt', 'r') as f:
#     cont = f.read()
# print(cont)

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

"""
合并表
   order_id  product_id  ...    days_since_prior_order  aisle
0         2       33120  ...                       8.0   eggs
1        26       33120  ...                       7.0   eggs
2       120       33120  ...                      10.0   eggs
3       327       33120  ...                       8.0   eggs
4       390       33120  ...                       9.0   eggs
5       537       33120  ...                       3.0   eggs
6       582       33120  ...                      10.0   eggs
7       608       33120  ...                      12.0   eggs
8       623       33120  ...                       3.0   eggs
9       689       33120  ...                       3.0   eggs

[10 rows x 14 columns]

交叉表
aisle    air fresheners candles  asian foods   ...    white wines  yogurt
user_id                                        ...                       
1                             0            0   ...              0       1
2                             0            3   ...              0      42
3                             0            0   ...              0       0
4                             0            0   ...              0       0
5                             0            2   ...              0       3
6                             0            0   ...              0       0
7                             0            0   ...              0       5
8                             0            1   ...              0       0
9                             0            0   ...              0      19
10                            0            1   ...              0       2

[10 rows x 134 columns]

降维，主成本分析
[[-2.42156587e+01  2.42942720e+00 -2.46636975e+00 ...  6.86800336e-01
   1.69439402e+00 -2.34323022e+00]
 [ 6.46320806e+00  3.67511165e+01  8.38255336e+00 ...  4.12121252e+00
   2.44689740e+00 -4.28348478e+00]
 [-7.99030162e+00  2.40438257e+00 -1.10300641e+01 ...  1.77534453e+00
  -4.44194030e-01  7.86665571e-01]
 ...
 [ 8.61143331e+00  7.70129866e+00  7.95240226e+00 ... -2.74252456e+00
   1.07112531e+00 -6.31925661e-02]
 [ 8.40862199e+01  2.04187340e+01  8.05410372e+00 ...  7.27554259e-01
   3.51339470e+00 -1.79079914e+01]
 [-1.39534562e+01  6.64621821e+00 -5.23030367e+00 ...  8.25329076e-01
   1.38230701e+00 -2.41942061e+00]]
   
data.shape
(206209, 27)
"""
