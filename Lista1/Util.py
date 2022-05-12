import numpy as np
import math

def split_training_test(x, y, percentage, shuffle=True, seed=0):
    if (shuffle):
        np.random.seed(seed)
        p = np.random.permutation(x.shape[0])
        x = x[p]
        y = y[p]

    i = math.floor(x.shape[0]*percentage)

    x_train = x[0:i,:]
    y_train = y[0:i]
    x_test = x[i:,:]
    y_test = y[i:]

    return x_train, y_train, x_test, y_test