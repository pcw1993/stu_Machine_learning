# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/7 2:39 PM
# software: PyCharm

# 使用gpu：

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一张图，包含op和tensor
# op：只要使用的TensorFlow的API定义的函数都是op
# 张量：Tensor：就指代数据
# g = tf.Graph()
#
# with g.as_default():
#     c = tf.constant(10.0)
#     print(c.graph)
#
#
#
# # 实现加法运算
# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# sum1 = tf.add(a,b)
# # 图，分配内存
# graph = tf.get_default_graph()
# print(graph)
# # print(sum1)
#
# plt = tf.placeholder(tf.float32, [None, 3])
# print(plt)
#
# # 一次只能运行一个图，
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:  # 看到程序在哪里运算的
#     print(sum1.eval())
#     print(sess.run(sum1))
#
#     print(a.graph)
#     print(a.shape)
#     print(plt.shape)
#     print(a.name)
#     print(a.op)


# # 一次只能运行一个图,可以在会话中指定
# with tf.Session(graph=g) as sess:
#     print(sess.run(c))


# 形状的概念，静态形状和动态形状
# 对于静态形状来说，一旦张量形状固定了，不能再次设置静态形状
# 动态形状可以去创建一个新的张量,改变时一定要注意元素数量要匹配

plt = tf.placeholder(tf.float32, [None, 2])

print(plt)

plt.set_shape([3,2])
print(plt)

# plt.set_shape([2,3])  # 不能再次修改

plt_reshape = tf.reshape(plt, [2,3])
print(plt_reshape)

with tf.Session() as sess:
    pass
