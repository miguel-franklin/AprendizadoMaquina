import abc
from abc import abstractmethod
class Model(metaclass=abc.ABCMeta):

    @abstractmethod()
    def predict(self):
        pass

def LinearRegression(Model):

    def __init__(self, w, x):
        self.w = w
        self.x = x

    def predict(self):
        return self.w @ self.x


def GD(model:Model, epochs=1000):
    for i in range(epochs):
        pass
