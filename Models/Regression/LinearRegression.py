import numpy as np
from Models import model


class LinearRegression(model.Model):

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def initial_weights(self):
        return np.zeros((1, self.x.shape[1]))

    def fit(self, x):
        ones = np.ones(self.x.shape[0])
        self.x = np.column_stack((ones, self.x[:, ]))

    def predict(self, x, weights):
        return x @ weights.T

    def OLS(self, y):
        return ((np.linalg.invp(self.x.T @ self.x) @ self.x.T) @ y).T
