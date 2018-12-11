# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/11 10:47 AM
# software: PyCharm

import tensorflow as tf


# name参数：在tensorboard使用的时候显示名字，可以让相同op的名字进行区分

# a = tf.constant([1,2,3,4,5])
a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.add(a,b, name='add')

var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0), name = 'variable')

print(a, var)

# 必须做一步显示初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)

    # 把程序的图结构写入事件文件，graph：把指定的图写进事件文件中
    filrwriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

    print(sess.run([a, var]))
