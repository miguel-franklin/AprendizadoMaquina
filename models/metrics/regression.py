import numpy as np

from .abstract import AbstractMeasure

class RMSE(AbstractMeasure):
    def measure(self, a, b):
        return np.sqrt(((a-b)**2).mean())

class MAE(AbstractMeasure):
    def measure(self, y1, y2):
        return np.mean(np.abs(y1 - y2))

class MRE(AbstractMeasure):
    def measure(self, y1, y2):
        return np.mean(np.abs(y1 - y2) / y1)

class MSE(AbstractMeasure):
    def measure(self, a, b):
        return ((a-b)**2).mean()
