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

    # 查看VAL_X
    SUB_1 = 12
    SUB_2 = 10
    plt.figure(num=1, figsize=(14, 8))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for i in xrange(4*7):
        # 可视化
        plt.subplot(4, 7, i+1)
        if (i < 7):
            plt.imshow(VAL_X[SUB_1*7+i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        elif (i < 14):
            plt.imshow(VAL_X[SUB_2*7+(i-7), :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        else:
            plt.imshow(VAL_X[i, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
    plt.savefig(OUTPUT_DIR + 'fig-VAL_X-1.pdf')
    plt.close()

    # 查看VAL_X
    plt.figure(num=1, figsize=(20, 20))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for i in xrange(100):
        # 可视化
        plt.subplot(10, 10, i + 1)
        plt.imshow(VAL_X[7 * i + 4, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
    plt.savefig(OUTPUT_DIR + 'fig-VAL_X-2.pdf')
    plt.close()

    # 查看VAL_X
    plt.figure(num=1, figsize=(40, 40))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for i in xrange(400):
        # 可视化
        plt.subplot(20, 20, i + 1)
        plt.imshow(VAL_X[7 * i + 4, :, :, :].squeeze().T, cmap='gray', vmin=0.0, vmax=255.0)
        plt.axis('off')
        plt.title(i, y=0.05)
    plt.savefig(OUTPUT_DIR + 'fig-VAL_X-3.pdf')
    plt.close()


if __name__ == '__main__':
    main()
