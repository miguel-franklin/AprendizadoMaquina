from abc import abstractmethod
import abc
import numpy as np


class Distance(metaclass=abc.ABCMeta):

    @abstractmethod
    def calculate(self):
        pass


class Euclidean(Distance):

    def fit(self, x):
        pass

    def calculate(self, j, i):
        """
        Distancia euclidiana dos parametros
        :param i: set de atributos i
        :param j: set de atributos j
        :return:
        """
        return np.sqrt(np.sum((j - i) ** 2))


class Mahalanobis(Distance):

    def __init__(self):
        self.cov_matrix = None

    def calculate(self, x, _x):
        passo1 = (x - _x)
        passo2 = self.cov_matrix
        passo3 = (x - _x).T

        passo_ = passo1 @ passo2 @ passo3
        if passo_.shape == ():
            return np.sqrt(passo_)
        else:
            return np.sqrt(np.diagonal(passo_))


    def fit(self, x):
        pinv = np.linalg.pinv(np.cov(x, rowvar=False))
        self.cov_matrix = pinv
