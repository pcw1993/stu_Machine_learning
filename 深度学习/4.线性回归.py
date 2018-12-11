# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/11 2:16 PM
# software: PyCharm
import os

import tensorflow as tf


# 训练参数问题
# 学习率和步数问题
# 添加权重参数，损失值在tensorboard观察的情况

# 定义命令行参数
# 1 首先定义哪些参数是需要在运行时候指定
# 2 程序中获取定义的命令行参数

tf.app.flags.DEFINE_integer('max_step', 100, '模型训练步数')
tf.app.flags.DEFINE_string('model_dir', './tmp/ckpt/model', '模型加载路径')

# 定义获取命令行参数名字
# python 4.线性回归.py --max_step=600 --model_dir='./tmp/ckpt/model'
FLAGS = tf.app.flags.FLAGS

def myregression():
    """
    自实现一个线性回归
    :return: None
    """
    # 变量作用域
    with tf.variable_scope('data'):
        # 准备数据， x 特征值[100, 10]  ,y 目标值 【100】
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope('model'):
        # 2，建立线性回归模型.1个权重，情歌偏置 y = xw + b
        # 随机给一个劝哄着那个和偏置的值吗让他去计算所示，然后再在当前的状态下优化
        # 用变量定义才能优化,trainable,指定这个变脸管是否跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name='w'))

        bias = tf.Variable(0.0, name='b')

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('loss'):
        # 3，建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope('optimizer'):
        # 4.梯度优化损失, learn_rate；0~1, 2,3
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 1.收集tensor
    tf.summary.scalar('losses', loss)
    tf.summary.histogram('weights', weight)

    # 2.定义合并tensor的op
    merged = tf.summary.merge_all()

    # 定义一个初始化op
    init_op = tf.global_variables_initializer()

    # 定义一个保存模型文件的实例op
    saver = tf.train.Saver()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化权重和偏置
        print('随机初始化参数权重为：%f， 偏置为： %f' % (weight.eval(), bias.eval()))

        # 建立事件文件
        filewrite = tf.summary.FileWriter('./tmp/summary/test', graph=sess.graph)

        # 加载模型，覆盖模型中随机定义的参数，从上次训练的参数结果开始训练
        if os.path.exists('./tmp/ckpt/checkpoint'):
            saver.restore(sess, FLAGS.model_dir)

        for i in range(FLAGS.max_step):
            # 运行优化
            sess.run((train_op))
            # 运行合并的tensor
            summary = sess.run(merged)
            filewrite.add_summary(summary, i)

            print('第%d次运行优化参数权重为：%f， 偏置为： %f' % (i + 1, weight.eval(), bias.eval()))
        saver.save(sess, FLAGS.model_dir)
    return None


if __name__ == '__main__':
    myregression()
