#!/usr/bin/python
# -*- coding:utf-8 -*-

import random
import time
import os
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
import tensorflow as tf
import h5py
import DATA
import matplotlib
matplotlib.use('Agg')  # 脱离X window使用
import matplotlib.pyplot as plt


def weight(shape, name=None):
    mlp = tf.get_variable(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    return mlp


def conv(mlp, c_out, name=None):
    c_in = mlp.get_shape()[3].value
    mlp = tf.nn.conv2d(mlp, weight([3, 3, c_in, c_out], name=name + 'w'), [1, 1, 1, 1], 'SAME')
    return mlp


def fc(mlp, c_out, name=None):
    c_in = mlp.get_shape()[1].value
    w = weight([c_in, c_out], name=name + 'w')
    mlp = tf.matmul(mlp, w)
    return mlp


def bn(mode, mlp):
    assert mode in ['train', 'val']
    mlp = tf.contrib.layers.batch_norm(mlp, is_training=mode == 'train', updates_collections=None, decay=0.9)
    return mlp


def pool(mlp):
    mlp = tf.nn.max_pool(mlp, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return mlp


def unpool(mlp):
    w = mlp.get_shape()[1].value
    h = mlp.get_shape()[2].value
    c = mlp.get_shape()[3].value
    mlp = tf.expand_dims(mlp, 2)
    mlp = tf.expand_dims(mlp, 4)
    mlp = tf.tile(mlp, [1, 1, 2, 1, 2, 1])
    mlp = mlp / 4.0

    mlp = tf.reshape(mlp, [-1, w*2, h*2, c])
    return mlp


def lrelu(mlp, leak=0.01, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * mlp + f2 * abs(mlp)


def build_graph_a(mode):
    print 'build graph a', mode
    assert mode in ['train', 'val']

    z = tf.placeholder(tf.float32, [None, 64, 64, 1])

    # 压缩表示
    with tf.variable_scope('X-1') as ns:
        mlp = z
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('CONV-2') as ns:
        mlp = conv(mlp, 16, name='conv21')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = conv(mlp, 16, name='conv22')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = pool(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('CONV-3') as ns:
        mlp = conv(mlp, 32, name='conv31')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = conv(mlp, 32, name='conv32')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = pool(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('CONV-4') as ns:
        mlp = conv(mlp, 64, name='conv41')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = conv(mlp, 64, name='conv42')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = pool(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('CONV-5') as ns:
        mlp = conv(mlp, 128, name='conv51')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = pool(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('CONV-6') as ns:
        mlp = conv(mlp, 256, name='conv61')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = pool(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('CONV-7') as ns:
        mlp = conv(mlp, 512, name='conv71')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = pool(mlp)
        print 'A', ns.name, mlp.get_shape()

        h2 = mlp

        return [z, h2]


def build_graph_b(mode):
    print 'build graph b', mode
    assert mode in ['train', 'val']

    x = tf.placeholder(tf.float32, [None, 64, 64, 1])
    y = tf.placeholder(tf.float32, [None, 7])
    h2 = tf.placeholder(tf.float32, [None, 1, 1, 512])
    mlp = tf.concat(3, [h2, tf.expand_dims(tf.expand_dims(y, 1), 1)])

    # 重构
    with tf.variable_scope('DECONV-10') as ns:
        mlp = unpool(mlp)
        mlp = conv(mlp, 256, name='conv101')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('DECONV-11') as ns:
        mlp = unpool(mlp)
        mlp = conv(mlp, 128, name='conv111')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('DECONV-12') as ns:
        mlp = unpool(mlp)
        mlp = conv(mlp, 64, name='conv121')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('DECONV-13') as ns:
        mlp = unpool(mlp)
        mlp = conv(mlp, 64, name='conv131')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = conv(mlp, 32, name='conv132')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('DECONV-14') as ns:
        mlp = unpool(mlp)
        mlp = conv(mlp, 32, name='conv141')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = conv(mlp, 16, name='conv142')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('DECONV-15') as ns:
        mlp = unpool(mlp)
        mlp = conv(mlp, 16, name='conv151')
        mlp = bn(mode, mlp)
        mlp = lrelu(mlp)
        mlp = conv(mlp, 1, name='conv152')
        mlp = bn(mode, mlp)
        # mlp = lrelu(mlp)
        print 'A', ns.name, mlp.get_shape()

    with tf.variable_scope('L_REC'):
        x_hat = mlp
        del mlp
        l = tf.nn.l2_loss(x - x_hat) / tf.cast(tf.shape(x)[0], tf.float32)

    if mode == 'train':
        with tf.variable_scope('OPT'):
            opt = tf.train.AdamOptimizer()
            train_op = opt.minimize(l)
            with tf.control_dependencies([train_op]):
                train_op = tf.add_check_numerics_ops()
    else:
        train_op = None

    return [x, y, h2, l, x_hat, train_op]


def main():
    # 常量
    OUTPUT_DIR = './100-NET12-VIS/'
    FOLD_FOR_VAL = 0

    # 加载数据集
    [TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z] = DATA.load_cv(FOLD_FOR_VAL)

    # 计算MEAN STD
    MEAN = np.mean(TRAIN_X, dtype=np.float32)
    STD = np.std(TRAIN_X, dtype=np.float32)
    print 'MEAN: ', MEAN
    print 'STD: ', STD

    # 设定四个用于插值的数据
    SUB_1 = 12
    SUB_2 = 10
    VAL_Z_1 = VAL_Z[7*SUB_1+4:7*SUB_1+4+1, :, :, :]
    VAL_Z_2 = VAL_Z[7*SUB_2+4:7*SUB_2+4+1, :, :, :]
    VAL_Y_1 = np.zeros([1, 7])
    VAL_Y_1[:, 3] = 1.0
    VAL_Y_2 = np.zeros([1, 7])
    VAL_Y_2[:, 1] = 1.0
    del TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z

    # 创建计算图
    with tf.Graph().as_default():
        # 为重现使用固定的随机数种子
        # 不同版本TF结果不同  同一版本下cpu/gpu结果相同
        SEED = 1
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        random.seed(SEED)

        # 创建网络
        with tf.variable_scope('GRAPH', reuse=None):
            [val_z, val_h2_a] = build_graph_a('val')
            [val_x, val_y, val_h2_b, val_l, val_x_hat, _] = build_graph_b('val')

        # 创建会话
        with tf.Session() as sess:
            # 加载快照
            print 'load snapshot'
            saver = tf.train.Saver(max_to_keep=1000000)
            saver.restore(sess, './100-NET2/snapshot-2000')

            # 设定四个用于插值的数据
            [VAL_H2_1] = sess.run([val_h2_a], feed_dict={val_z: (VAL_Z_1 - MEAN) / STD})
            [VAL_H2_2] = sess.run([val_h2_a], feed_dict={val_z: (VAL_Z_2 - MEAN) / STD})

            # 验证
            # 保留验证集尾部
            # 准备MB
            D = 5
            plt.figure(num=1, figsize=(D*2, D*2))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            for i in xrange(D):
                for j in xrange(D):
                    y_ratio = i / (D-1.0)
                    h2_ratio = j / (D-1.0)
                    val_y_val = VAL_Y_1 * (1.0 - y_ratio) + VAL_Y_2 * y_ratio
                    val_h2_b_val = VAL_H2_1 * (1.0 - h2_ratio) + VAL_H2_2 * h2_ratio

                    # run
                    [val_x_hat_val] = sess.run([val_x_hat], feed_dict={val_y: val_y_val, val_h2_b: val_h2_b_val})

                    # 可视化
                    plt.subplot(D, D, i * D + j + 1)
                    plt.imshow((val_x_hat_val[0, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                    plt.axis('off')
                    # plt.title("%.1f %.1f" % (y_ratio, h2_ratio), y=0.1)

            plt.savefig(OUTPUT_DIR + 'fig-interp-h2+y.pdf')
            plt.close()


if __name__ == '__main__':
    main()
