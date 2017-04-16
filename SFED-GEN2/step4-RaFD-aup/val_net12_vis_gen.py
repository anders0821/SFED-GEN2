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

    # GEN
    SUB = 12
    TRAIN_Z = TRAIN_Z[7*(7*SUB+3):7*(7*SUB+3)+1, :, :, :]
    TRAIN_Z = np.tile(TRAIN_Z, [7, 1, 1, 1])
    TRAIN_Y = np.diag(np.ones(7, np.uint8))
    del TRAIN_X
    VAL_Z = VAL_Z[7*(7*SUB+3):7*(7*SUB+3)+1, :, :, :]
    VAL_Z = np.tile(VAL_Z, [7, 1, 1, 1])
    VAL_Y = np.diag(np.ones(7, np.uint8))
    del VAL_X

    # 创建计算图
    with tf.Graph().as_default():
        # 创建网络
        with tf.variable_scope('GRAPH', reuse=None):
            [_, val_y, val_z, _, val_x_hat, _] = train_val_net2.build_graph('val')

        # 创建会话
        with tf.Session() as sess:
            # 加载快照
            saver = tf.train.Saver(max_to_keep=1000000)
            print 'load snapshot'
            saver.restore(sess, './100-NET2/snapshot-200')

            # 验证
            # 保留验证集尾部
            VAL_X_HAT = np.zeros(np.shape(VAL_Z))
            ITER_COUNT = ((VAL_Z.shape[0] - 1) / MB) + 1
            for itr in xrange(ITER_COUNT):
                # 准备MB
                mb = min(itr * MB + MB, VAL_Z.shape[0]) - itr * MB
                val_y_val = VAL_Y[itr * MB:itr * MB + mb, :]
                val_z_val = VAL_Z[itr * MB:itr * MB + mb, :, :, :]
                val_z_val = (val_z_val - MEAN) / STD

                # run
                [val_x_hat_val] = \
                    sess.run([val_x_hat], feed_dict={val_y: val_y_val, val_z: val_z_val})
                VAL_X_HAT[itr * MB:itr * MB + mb, :, :, :] = val_x_hat_val * STD + MEAN

            # 验证 on trainnin set
            # 保留验证集尾部
            TRAIN_X_HAT = np.zeros(np.shape(TRAIN_Z))
            ITER_COUNT = ((TRAIN_Z.shape[0] - 1) / MB) + 1
            for itr in xrange(ITER_COUNT):
                # 准备MB
                mb = min(itr * MB + MB, TRAIN_Z.shape[0]) - itr * MB
                val_y_val = TRAIN_Y[itr * MB:itr * MB + mb, :]  # GEN
                val_z_val = TRAIN_Z[itr * MB:itr * MB + mb, :, :, :]
                val_z_val = (val_z_val - MEAN) / STD

                # run
                [val_x_hat_val] = \
                    sess.run([val_x_hat], feed_dict={val_y: val_y_val, val_z: val_z_val})
                TRAIN_X_HAT[itr * MB:itr * MB + mb, :, :, :] = val_x_hat_val * STD + MEAN

    # 可视化
    plt.figure(num=1, figsize=(24, 13.5))
    for i in xrange(7):
        plt.subplot(7, 8, i * 8 + 2)
        plt.imshow(TRAIN_Z[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(TRAIN_Y[i, :])

        plt.subplot(7, 8, i * 8 + 4)
        plt.imshow(TRAIN_X_HAT[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')

        plt.subplot(7, 8, i * 8 + 6)
        plt.imshow(VAL_Z[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(VAL_Y[i, :])

        plt.subplot(7, 8, i * 8 + 8)
        plt.imshow(VAL_X_HAT[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')

    plt.savefig(OUTPUT_DIR + 'fig-gen.pdf')
    plt.close()


if __name__ == '__main__':
    main()
