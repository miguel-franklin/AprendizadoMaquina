import abc
from abc import abstractmethod


class Model(metaclass=abc.ABCMeta):

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def predict(self, x, weights):
        pass

    @abstractmethod
    def predict_train(self, x, weights):
        pass

    @abstractmethod
    def initial_weights(self, x):
        pass
