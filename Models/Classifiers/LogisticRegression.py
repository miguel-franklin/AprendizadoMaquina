import numpy as np
from Models.model import Model

class LogisticRegression(Model):

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def sigmoid(self, x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig


    def predict(self, inputs):
        if self.trained:
            return np.round(super().predict(inputs))
        else:
            return super().predict(inputs)

    def predict_proba(self, inputs):
        return super().predict(inputs)



    def initial_weights(self):
        return np.zeros((1, self.x.shape[1]))

    def fit(self, x):
        ones = np.ones(self.x.shape[0])
        self.x = np.column_stack((ones, self.x[:, ]))

    def predict(self, x, weights):
        if self.trained:
    return np.round(super().predict(inputs))
    else:
    return super().predict(inputs)

return x @ weights.T

    def OLS(self, y):
        return ((np.linalg.invp(self.x.T @ self.x) @ self.x.T) @ y).T
