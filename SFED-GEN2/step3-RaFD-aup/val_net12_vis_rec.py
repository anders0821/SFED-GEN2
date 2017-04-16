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

import train_val_net1
import train_val_net2


def main():
    # 常量
    OUTPUT_DIR = './100-NET12-VIS/'
    MB = 25
    FOLD_FOR_VAL = 0

    # 加载数据集
    [TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z] = DATA.load_cv(FOLD_FOR_VAL)

    # 计算MEAN STD
    MEAN = np.mean(TRAIN_X, dtype=np.float32)
    STD = np.std(TRAIN_X, dtype=np.float32)
    print 'MEAN: ', MEAN
    print 'STD: ', STD

    # REC
    SUB = 12
    TRAIN_X = TRAIN_X[7*SUB:7*SUB+7, :, :, :]
    TRAIN_Y = TRAIN_Y[7*SUB:7*SUB+7, :]
    TRAIN_Z = TRAIN_Z[7*SUB:7*SUB+7, :, :, :]
    VAL_X = VAL_X[7*SUB:7*SUB+7, :, :, :]
    VAL_Y = VAL_Y[7*SUB:7*SUB+7, :]
    VAL_Z = VAL_Z[7*SUB:7*SUB+7, :, :, :]

    # 创建计算图
    with tf.Graph().as_default():
        # 创建网络
        with tf.variable_scope('GRAPH', reuse=None):
            [val_x, val_y, val_z, val_l_rec, val_l_cls, val_l, val_z_hat, val_y_hat, val_acc, _] = train_val_net1.build_graph('val')

        # 创建会话
        sess_config = tf.ConfigProto()
        with tf.Session(config=sess_config) as sess:
            # 加载快照
            saver = tf.train.Saver(max_to_keep=1000000)
            print 'load snapshot'
            saver.restore(sess, './100-NET1/snapshot-2000')

            # 验证
            # 保留验证集尾部
            VAL_Z_HAT = np.zeros(np.shape(VAL_Z))
            VAL_Y_HAT = np.zeros(np.shape(VAL_Y))
            mean_val_l_rec = 0.0
            mean_val_l_cls = 0.0
            mean_val_l = 0.0
            mean_val_acc = 0.0
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
                [val_l_rec_val, val_l_cls_val, val_l_val, val_z_hat_val, val_y_hat_val, val_acc_val] = \
                    sess.run([val_l_rec, val_l_cls, val_l, val_z_hat, val_y_hat, val_acc], feed_dict={val_x: val_x_val, val_y: val_y_val, val_z: val_z_val})
                mean_val_l_rec += val_l_rec_val * mb
                mean_val_l_cls += val_l_cls_val * mb
                mean_val_l += val_l_val * mb
                mean_val_acc += val_acc_val * mb
                mean_val_count += mb
                VAL_Z_HAT[itr * MB:itr * MB + mb, :, :, :] = val_z_hat_val * STD + MEAN
                VAL_Y_HAT[itr * MB:itr * MB + mb, :] = val_y_hat_val
                # print val_l_rec_val, val_l_cls_val, val_acc_val
            print 'mean_val_l_rec %g, mean_val_l_cls %g, mean_val_l %g, mean_val_acc %g' % \
                  (mean_val_l_rec / mean_val_count, mean_val_l_cls / mean_val_count, mean_val_l / mean_val_count, mean_val_acc / mean_val_count)

            # 验证 on training set
            # 保留验证集尾部
            TRAIN_Z_HAT = np.zeros(np.shape(TRAIN_Z))
            TRAIN_Y_HAT = np.zeros(np.shape(TRAIN_Y))
            mean_val_l_rec = 0.0
            mean_val_l_cls = 0.0
            mean_val_l = 0.0
            mean_val_acc = 0.0
            mean_val_count = 0
            ITER_COUNT = ((TRAIN_X.shape[0] - 1) / MB) + 1
            for itr in xrange(ITER_COUNT):
                # 准备MB
                mb = min(itr * MB + MB, TRAIN_X.shape[0]) - itr * MB
                val_x_val = TRAIN_X[itr * MB:itr * MB + mb, :, :, :]
                val_y_val = TRAIN_Y[itr * MB:itr * MB + mb, :]
                val_z_val = TRAIN_Z[itr * MB:itr * MB + mb, :, :, :]
                val_x_val = (val_x_val - MEAN) / STD
                val_z_val = (val_z_val - MEAN) / STD

                # run
                [val_l_rec_val, val_l_cls_val, val_l_val, val_z_hat_val, val_y_hat_val, val_acc_val] = \
                    sess.run([val_l_rec, val_l_cls, val_l, val_z_hat, val_y_hat, val_acc], feed_dict={val_x: val_x_val, val_y: val_y_val, val_z: val_z_val})
                mean_val_l_rec += val_l_rec_val * mb
                mean_val_l_cls += val_l_cls_val * mb
                mean_val_l += val_l_val * mb
                mean_val_acc += val_acc_val * mb
                mean_val_count += mb
                TRAIN_Z_HAT[itr * MB:itr * MB + mb, :, :, :] = val_z_hat_val * STD + MEAN
                TRAIN_Y_HAT[itr * MB:itr * MB + mb, :] = val_y_hat_val
                # print val_l_rec_val, val_l_cls_val, val_acc_val
            print 'mean_val_l_rec %g, mean_val_l_cls %g, mean_val_l %g, mean_val_acc %g' % \
                  (mean_val_l_rec / mean_val_count, mean_val_l_cls / mean_val_count, mean_val_l / mean_val_count, mean_val_acc / mean_val_count)

    # 创建计算图
    with tf.Graph().as_default():
        # 创建网络
        with tf.variable_scope('GRAPH', reuse=None):
            [val_x, val_y, val_z, val_l, val_x_hat, _] = train_val_net2.build_graph('val')

        # 创建会话
        with tf.Session() as sess:
            # 加载快照
            saver = tf.train.Saver(max_to_keep=1000000)
            print 'load snapshot'
            saver.restore(sess, './100-NET2/snapshot-2000')

            # 验证
            # 保留验证集尾部
            VAL_X_HAT = np.zeros(np.shape(VAL_X))
            mean_val_l = 0.0
            mean_val_count = 0
            ITER_COUNT = ((VAL_X.shape[0] - 1) / MB) + 1
            for itr in xrange(ITER_COUNT):
                # 准备MB
                mb = min(itr * MB + MB, VAL_X.shape[0]) - itr * MB
                val_x_val = VAL_X[itr * MB:itr * MB + mb, :, :, :]
                val_y_val = VAL_Y_HAT[itr * MB:itr * MB + mb, :]  # REC
                val_z_val = VAL_Z_HAT[itr * MB:itr * MB + mb, :, :, :]
                val_x_val = (val_x_val - MEAN) / STD
                val_z_val = (val_z_val - MEAN) / STD

                # run
                [val_l_val, val_x_hat_val] = \
                    sess.run([val_l, val_x_hat], feed_dict={val_x: val_x_val, val_y: val_y_val, val_z: val_z_val})
                mean_val_l += val_l_val * mb
                mean_val_count += mb
                VAL_X_HAT[itr * MB:itr * MB + mb, :, :, :] = val_x_hat_val * STD + MEAN
                # print val_l_val
            print 'mean_val_l %g' % (mean_val_l / mean_val_count)

            # 验证 on trainnin set
            # 保留验证集尾部
            TRAIN_X_HAT = np.zeros(np.shape(TRAIN_X))
            mean_val_l = 0.0
            mean_val_count = 0
            ITER_COUNT = ((TRAIN_X.shape[0] - 1) / MB) + 1
            for itr in xrange(ITER_COUNT):
                # 准备MB
                mb = min(itr * MB + MB, TRAIN_X.shape[0]) - itr * MB
                val_x_val = TRAIN_X[itr * MB:itr * MB + mb, :, :, :]
                val_y_val = TRAIN_Y_HAT[itr * MB:itr * MB + mb, :]  # REC
                val_z_val = TRAIN_Z_HAT[itr * MB:itr * MB + mb, :, :, :]
                val_x_val = (val_x_val - MEAN) / STD
                val_z_val = (val_z_val - MEAN) / STD

                # run
                [val_l_val, val_x_hat_val] = \
                    sess.run([val_l, val_x_hat], feed_dict={val_x: val_x_val, val_y: val_y_val, val_z: val_z_val})
                mean_val_l += val_l_val * mb
                mean_val_count += mb
                TRAIN_X_HAT[itr * MB:itr * MB + mb, :, :, :] = val_x_hat_val * STD + MEAN
                # print val_l_val
            print 'mean_val_l %g' % (mean_val_l / mean_val_count)

    # 可视化
    plt.figure(num=1, figsize=(24, 13.5))
    for i in xrange(7):
        plt.subplot(7, 8, i * 8 + 1)
        plt.imshow(TRAIN_X[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')

        plt.subplot(7, 8, i * 8 + 2)
        plt.imshow(TRAIN_Z[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(TRAIN_Y[i, :])

        plt.subplot(7, 8, i * 8 + 3)
        plt.imshow(TRAIN_Z_HAT[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(TRAIN_Y_HAT[i, :])

        plt.subplot(7, 8, i * 8 + 4)
        plt.imshow(TRAIN_X_HAT[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')

        plt.subplot(7, 8, i * 8 + 5)
        plt.imshow(VAL_X[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')

        plt.subplot(7, 8, i * 8 + 6)
        plt.imshow(VAL_Z[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(VAL_Y[i, :])

        plt.subplot(7, 8, i * 8 + 7)
        plt.imshow(VAL_Z_HAT[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(VAL_Y_HAT[i, :])

        plt.subplot(7, 8, i * 8 + 8)
        plt.imshow(VAL_X_HAT[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')

    plt.savefig(OUTPUT_DIR + 'fig-rec.pdf')
    plt.close()


if __name__ == '__main__':
    main()
