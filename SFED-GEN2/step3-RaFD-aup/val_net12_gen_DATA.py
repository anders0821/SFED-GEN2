# -*- coding:utf-8 -*-

import numpy as np
import h5py


def load_cv(FOLD_FOR_TEST):
    # FN = '../SFED2-150.mat'
    # N = 150

    # FN = '../SFED2-1500.mat'
    # N = 1500

    # FN = '../SFED2-15000.mat'
    # N = 15000

    FN = '../RaFD.mat'
    N = 67

    # X
    h5 = h5py.File(FN, 'r')
    X = h5['X'][:]
    assert(X.shape == (N, 7, 64, 64))
    Z = X
    X = X.reshape([N, 1, 7, 64, 64, 1])
    X = np.tile(X, [1, 7, 1, 1, 1, 1])
    X = X.reshape([N*7*7, 64, 64, 1])

    # Z
    Z = Z.reshape([N, 7, 1, 64, 64, 1])
    Z = np.tile(Z, [1, 1, 7, 1, 1, 1])
    Z = Z.reshape([N*7*7, 64, 64, 1])

    # Y
    Y = np.zeros([N, 7, 7, 7], np.uint8)
    for i in xrange(N):
        for j in xrange(7):
            for k in xrange(7):
                Y[i, j, k, k] = 1
    Y = Y.reshape([N*7*7, 7])

    # CV分集
    FOLD = h5['FOLD'][:]
    assert(FOLD.shape == (1, N))
    FOLD = np.tile(FOLD.T, [1, 7*7]).flatten()
    TRAIN_X = X[FOLD != FOLD_FOR_TEST]
    VAL_X = X[FOLD == FOLD_FOR_TEST]
    del X
    TRAIN_Y = Y[FOLD != FOLD_FOR_TEST]
    VAL_Y = Y[FOLD == FOLD_FOR_TEST]
    del Y
    TRAIN_Z = Z[FOLD != FOLD_FOR_TEST]
    VAL_Z = Z[FOLD == FOLD_FOR_TEST]
    del Z

    # print shape
    print 'load cv'
    print 'TRAIN_X.shape', TRAIN_X.shape
    print 'TRAIN_Y.shape', TRAIN_Y.shape
    print 'TRAIN_Z.shape', TRAIN_Z.shape
    print 'VAL_X.shape', VAL_X.shape
    print 'VAL_Y.shape', VAL_Y.shape
    print 'VAL_Z.shape', VAL_Z.shape

    return [TRAIN_X, TRAIN_Y, TRAIN_Z, VAL_X, VAL_Y, VAL_Z]
