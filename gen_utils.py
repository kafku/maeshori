# coding: utf-8

import numpy as np
import math

def _resize_batch(X, y, factor):
    """
    """
    # get indices
    if isinstance(X, dict):
        key = next(iter(X.keys()))
        data_size = X[key].shape[0]
    else:
        data_size = X.shape[0]

    sample_size = math.ceil(data_size / factor) * factor - data_size
    sample_idx = np.random.choice(data_size, sample_size, replace=False)

    if len(sample_idx) == 0:
        return X, y

    # get indices
    extended_idx = np.concatenate([np.arange(data_size), sample_idx])
    np.random.shuffle(extended_idx)

    # resize
    if isinstance(X, dict):
        X_train = dict()
        for key in X.keys():
            X_train[key] = X[key][extended_idx, ]
    else:
        X_train = X[extended_idx, ]

    if isinstance(y, dict):
        y_train = dict()
        for key in y.keys():
            y_train[key] = y[key][extended_idx,]
    else:
        y_train = y[extended_idx,]

    return X_train, y_train

def stack_batch(generator, stack_size):
    """
    Args:
        generator: generator that returns X, y
        stack_size: number of batches to be stacked
    Returns:
        X_train, y_train
    """
    while True:
        X_train = None
        y_train = None
        X_train, y_train = next(generator)

        # stack batches
        for _ in range(stack_size - 1):
            X, y = next(generator)

            if isinstance(X_train, dict):
                for key in X_train.keys():
                    X_train[key] = np.concatenate([X_train[key], X[key]])
            else:
                X_train = np.concatenate([X_train, X])

            if isinstance(y_train, dict):
                for key in y_train.keys():
                    y_train[key] = np.concate([y_train[key], y[key]])
            else:
                y_train = np.concatenate([y_train, y])

        X_train, y_train = _resize_batch(X_train, y_train, stack_size)
        yield X_train, y_train
