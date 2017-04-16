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


def main():
    # 常量
    OUTPUT_DIR = './100-NET12-VIS/'
    FOLD_FOR_VAL = 0

    # 加载数据集
    [TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z] = DATA.load_cv(FOLD_FOR_VAL)

    # 设定四个用于插值的数据
    SUB_1 = 12
    SUB_2 = 10
    VAL_1_1 = VAL_X[7*SUB_1+3, :, :, 0]
    VAL_1_2 = VAL_X[7*SUB_1+1, :, :, 0]
    VAL_2_1 = VAL_X[7*SUB_2+3, :, :, 0]
    VAL_2_2 = VAL_X[7*SUB_2+1, :, :, 0]
    del TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z

    D = 5
    plt.figure(num=1, figsize=(D * 2, D * 2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for i in xrange(D):
        for j in xrange(D):
            y_ratio = i / (D - 1.0)
            z_ratio = j / (D - 1.0)
            val_1 = VAL_1_1 * (1.0 - y_ratio) + VAL_1_2 * y_ratio
            val_2 = VAL_2_1 * (1.0 - y_ratio) + VAL_2_2 * y_ratio
            val = val_1 * (1.0 - z_ratio) + val_2 * z_ratio

            # 可视化
            plt.subplot(D, D, i * D + j + 1)
            plt.imshow(val.T, cmap='gray', vmin=0.0, vmax=255.0)
            plt.axis('off')
            # plt.title("%.1f %.1f" % (y_ratio, z_ratio), y=0.1)

    plt.savefig(OUTPUT_DIR + 'fig-interp-x-4.pdf')
    plt.close()

    D = 5
    plt.figure(num=1, figsize=(D*2, D*2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for i in xrange(D):
        for j in xrange(D):
            # 可视化
            plt.subplot(D, D, i * D + j + 1)
            plt.imshow(np.zeros([64,64]), cmap='gray', vmin=0.0, vmax=255.0)
            plt.axis('off')

    for i in xrange(D):
        ratio = i / (D-1.0)
        val = VAL_1_1 * (1.0 - ratio) + VAL_2_2 * ratio

        # 可视化
        plt.subplot(D, D, i * D + i + 1)
        plt.imshow(val.T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        # plt.title("%.1f %.1f" % (y_ratio, z_ratio), y=0.1)

    plt.savefig(OUTPUT_DIR + 'fig-interp-x-2.pdf')
    plt.close()


if __name__ == '__main__':
    main()
