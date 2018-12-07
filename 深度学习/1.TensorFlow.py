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
g = tf.Graph()

with g.as_default():
    c = tf.constant(10.0)
    print(c.graph)



# 实现加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)
# 图，分配内存
graph = tf.get_default_graph()
print(graph)
# print(sum1)

# 一次只能运行一个图，看到程序在哪里运算的
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sum1.eval())
    print(sess.run(sum1))


# 一次只能运行一个图,可以在会话中指定
with tf.Session(graph=g) as sess:
    print(sess.run(c))