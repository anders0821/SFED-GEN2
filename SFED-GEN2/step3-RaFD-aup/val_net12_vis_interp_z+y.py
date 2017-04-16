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
            [val_x, val_y, val_z, val_l, val_x_hat, _] = train_val_net2.build_graph('val')

        # 创建会话
        with tf.Session() as sess:
            # 加载快照
            print 'load snapshot'
            saver = tf.train.Saver(max_to_keep=1000000)
            saver.restore(sess, './100-NET2/snapshot-2000')

            # 验证
            # 保留验证集尾部
            # 准备MB
            D = 5
            plt.figure(num=1, figsize=(D*2, D*2))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            for i in xrange(D):
                for j in xrange(D):
                    y_ratio = i / (D-1.0)
                    z_ratio = j / (D-1.0)
                    val_y_val = VAL_Y_1 * (1.0 - y_ratio) + VAL_Y_2 * y_ratio
                    val_z_val = VAL_Z_1 * (1.0 - z_ratio) + VAL_Z_2 * z_ratio
                    val_z_val = (val_z_val - MEAN) / STD

                    # run
                    [val_x_hat_val] = sess.run([val_x_hat], feed_dict={val_y: val_y_val, val_z: val_z_val})

                    # 可视化
                    plt.subplot(D, D, i * D + j + 1)
                    plt.imshow((val_x_hat_val[0, :, :, :].squeeze().T * STD) + MEAN, cmap='gray', vmin=0.0, vmax=255.0)
                    plt.axis('off')
                    # plt.title("%.1f %.1f" % (y_ratio, z_ratio), y=0.1)

            plt.savefig(OUTPUT_DIR + 'fig-interp-z+y.pdf')
            plt.close()


if __name__ == '__main__':
    main()
