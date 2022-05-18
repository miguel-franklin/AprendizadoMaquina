import numpy as np
import abc
from abc import abstractmethod
from models.metrics.metrics_factory import get_classification_metrics
from models.normalization import MinMaxScaler

from models import model


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
        sum = np.sum((i - j) ** 2, axis=1)
        return np.sqrt(sum)


class Mahalanobis1(Distance):

    def __init__(self):
        self.cov_matrix = None

    def calculate(self, x, _x):
        passo1 = (x - _x)
        passo2 = self.cov_matrix
        passo3 = (x - _x).T

        return np.sqrt(np.diagonal(passo1 @ passo2 @ passo3))

    def fit(self, x):
        pinv = np.linalg.pinv(np.cov(x, rowvar=False))
        self.cov_matrix = pinv


class Kmeans(model.Model):

    def predict_train(self, x, weights):
        pass

    def initial_weights(self, x):
        pass

    def __init__(self, k=3, distance='Euclidean', normalization=None, metrics='F1'):

        if distance == 'Euclidean':
            self.distance = Euclidean()
        elif distance == 'Mahalanobis':
            self.distance = Mahalanobis1()

        self.y = None
        self.x = None
        self.k = k
        self.normalization = normalization
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.metrics = get_classification_metrics(metrics)

    def fit(self, x, y, training=None):
        self.y = y
        self.x = self.x_scaler.fit(x).transform(x)
        self.distance.fit(self.x)

    def predict(self, x_test):
        prediction = []
        x_test = self.x_scaler.transform(x_test)
        for xi in x_test:
            distances = self.distance.calculate(self.x, xi)
            closest_ks = distances.argsort()[:self.k]
            k_classes = self.y[closest_ks]
            prediction.append(np.bincount(k_classes.astype(int)).argmax())

        return prediction
