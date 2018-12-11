# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/11 10:38 AM
# software: PyCharm

# 变量op

import tensorflow as tf


a = tf.constant([1,2,3,4,5])

var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0))

# print(a, var)

# 必须做一步显示初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)

    print(sess.run([a, var]))
