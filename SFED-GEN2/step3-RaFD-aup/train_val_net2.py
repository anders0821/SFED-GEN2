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


def build_graph(mode):
    print 'build graph', mode
    assert mode in ['train', 'val']

    x = tf.placeholder(tf.float32, [None, 64, 64, 1])
    y = tf.placeholder(tf.float32, [None, 7])
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
        mlp = tf.concat(3, [mlp, tf.expand_dims(tf.expand_dims(y, 1), 1)])
        print 'A', ns.name, mlp.get_shape()

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

    return [x, y, z, l, x_hat, train_op]


def main():
    # 常量
    OUTPUT_DIR = './100-NET2/'
    MB = 25
    FOLD_FOR_VAL = 0
    SNAPSHOT_RESUME_FROM = 0
    EPOCH_MAX = 2000
    SNAPSHOT_INTERVAL = 200

    # 加载数据集
    [TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z] = DATA.load_cv(FOLD_FOR_VAL)

    # 计算MEAN STD
    MEAN = np.mean(TRAIN_X, dtype=np.float32)
    STD = np.std(TRAIN_X, dtype=np.float32)
    print 'MEAN: ', MEAN
    print 'STD: ', STD

    '''
    # 缩短数据集 只计算少量数据 用于测试及可视化
    TRAIN_X = TRAIN_X[:MB, :, :, :]
    TRAIN_Y = TRAIN_Y[:MB, :]
    TRAIN_Z = TRAIN_Z[:MB, :, :, :]
    VAL_X = VAL_X[:MB, :, :, :]
    VAL_Y = VAL_Y[:MB, :]
    VAL_Z = VAL_Z[:MB, :, :, :]
    '''

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
            [train_x, train_y, train_z, train_l, train_x_hat, train_op] = build_graph('train')
        with tf.variable_scope('GRAPH', reuse=True):
            [val_x, val_y, val_z, val_l, val_x_hat, _] = build_graph('val')

        # 创建会话
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=1000000)

            # 初始化变量或加载快照
            if SNAPSHOT_RESUME_FROM == 0:
                print 'init vars'
                tf.global_variables_initializer().run()
            else:
                print 'load snapshot'
                saver.restore(sess, OUTPUT_DIR + 'snapshot-' + str(SNAPSHOT_RESUME_FROM))

            # 训练循环
            # 1 ~ EPOCH_MAX 或 SNAPSHOT_RESUME_FROM+1 ~ EPOCH_MAX
            for epoch in xrange(SNAPSHOT_RESUME_FROM + 1, EPOCH_MAX + 1):
                print '---------- epoch %d ----------' % epoch
                t = time.time()

                # 打乱训练集
                idx = np.random.permutation(TRAIN_X.shape[0])
                TRAIN_X = TRAIN_X[idx, :, :, :]
                TRAIN_Y = TRAIN_Y[idx, :]
                TRAIN_Z = TRAIN_Z[idx, :]

                # 训练
                # 抛弃训练集尾部 担心变化的MB会影响ADAM BATCHNORM等计算
                mean_train_l = 0.0
                mean_train_count = 0
                ITER_COUNT = TRAIN_X.shape[0] / MB
                for itr in xrange(ITER_COUNT):
                    # 准备MB
                    train_x_val = TRAIN_X[itr * MB:itr * MB + MB, :, :, :]
                    train_y_val = TRAIN_Y[itr * MB:itr * MB + MB, :]
                    train_z_val = TRAIN_Z[itr * MB:itr * MB + MB, :]
                    train_x_val = (train_x_val - MEAN) / STD
                    train_z_val = (train_z_val - MEAN) / STD

                    '''
                    # 可视化MB
                    plt.figure(num=1, figsize=(24, 13.5))
                    for i in xrange(10):
                        plt.subplot(10, 2, i*2+1)
                        plt.imshow((train_x_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')
                        plt.subplot(10, 2, i*2+2)
                        plt.imshow((train_z_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')
                        plt.title(train_y_val[i, :])
                    plt.savefig(OUTPUT_DIR + 'mb-' + str(epoch) + '.pdf')
                    plt.close()
                    return
                    '''

                    # run
                    [_, train_l_val, train_x_hat_val] = \
                        sess.run([train_op, train_l, train_x_hat], feed_dict={train_x: train_x_val, train_y: train_y_val, train_z: train_z_val})
                    mean_train_l += train_l_val * MB
                    mean_train_count += MB
                    # print train_l_val
                print 'mean_train_l %g' % (mean_train_l / mean_train_count)

                # 验证
                # 保留验证集尾部
                mean_val_l = 0.0
                mean_val_count = 0
                ITER_COUNT = ((VAL_X.shape[0] - 1) / MB) + 1
                for itr in xrange(ITER_COUNT):
                    # 准备MB
                    mb = min(itr * MB + MB, VAL_X.shape[0]) - itr * MB
                    val_x_val = VAL_X[itr * MB:itr * MB + mb, :, :, :]
                    val_y_val = VAL_Y[itr * MB:itr * MB + mb, :]
                    val_z_val = VAL_Z[itr * MB:itr * MB + mb, :, :, :]
                    val_x_val = (val_x_val - MEAN) / STD
                    val_z_val = (val_z_val - MEAN) / STD

                    # run
                    [val_l_val, val_x_hat_val] = \
                        sess.run([val_l, val_x_hat], feed_dict={val_x: val_x_val, val_y: val_y_val, val_z: val_z_val})
                    mean_val_l += val_l_val * mb
                    mean_val_count += mb
                    # print val_l_val
                print 'mean_val_l %g' % (mean_val_l / mean_val_count)

                # 可视化
                if (epoch % SNAPSHOT_INTERVAL) == 0:
                    plt.figure(num=1, figsize=(24, 13.5))
                    for i in xrange(3):
                        plt.subplot(3, 6, i * 6 + 1)
                        plt.imshow((train_x_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')

                        plt.subplot(3, 6, i * 6 + 2)
                        plt.imshow((train_z_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')
                        plt.title(train_y_val[i, :])

                        plt.subplot(3, 6, i * 6 + 3)
                        plt.imshow((train_x_hat_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')

                        plt.subplot(3, 6, i * 6 + 4)
                        plt.imshow((val_x_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')

                        plt.subplot(3, 6, i * 6 + 5)
                        plt.imshow((val_z_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')
                        plt.title(val_y_val[i, :])

                        plt.subplot(3, 6, i * 6 + 6)
                        plt.imshow((val_x_hat_val[i, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                        plt.axis('off')
                    plt.savefig(OUTPUT_DIR+'fig-'+str(epoch)+'.pdf')
                    plt.close()

                # 计划的save snapshot
                if (epoch % SNAPSHOT_INTERVAL) == 0:
                    saver.save(sess, OUTPUT_DIR+'snapshot-'+str(epoch), write_meta_graph=False)
                    print 'save snapshot'

                print 't %g' % (time.time() - t)


if __name__ == '__main__':
    main()
