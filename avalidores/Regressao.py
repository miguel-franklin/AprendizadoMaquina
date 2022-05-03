import numpy as np


def RMSE(y, _y):
    """
    root mean squared error
    :return:
    """
    return np.sqrt(((y - _y) ** 2).mean())


def MAE(y, _y):
    """
    mean absolute error
    :return:
    """
    pass


def MRE(y, _y):
    """
    mean relative error
    :return:
    """


def SE():
    """
    standard error
    :return:
    """


def R2():
    """
    coeficiente de determniação R2
    Tem uma boa performance para modelos lineares, e não trabalha muito bem
    para modelos não-lineares
    :return:
    """
    pass
