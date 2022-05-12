import numpy as np


class PolynomModel:
    def __init__(self, grandeza=2):
        self.grandeza = grandeza

    def normalizacao_min_max(self, z):
        min = np.min(z, axis=0)
        max = np.max(z, axis=0)
        return (z - min) / (max - min), min, max

    def denorm_x(self, z, min, max):
        return (z * (max - min)) + min

    def polynom(self, x):
        pol = 2
        stacks = []
        for _ in range(self.grandeza-1):
            stacks.append(x ** pol)
            pol += 1
        for i in range(len(stacks)):
            x = np.column_stack((x, stacks[i]))
        return x

    def fit(self, X, y):
        y = y.reshape(y.shape[0], 1)
        X = self.polynom(X)
        X, self.x_min, self.x_max = self.normalizacao_min_max(X)
        y, self.y_min, self.y_max = self.normalizacao_min_max(y)
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X[:,]))
        self.x = X
        self.y = y

    def stack_ones(self, X):
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X[:,]))
        return X


    def regressao_polinomial(self, X, W):
        return X@W.T

    def OLS(self, regularization=0.0):
        if regularization is 0.0:
            return ((np.linalg.pinv(self.x.T @ self.x) @ self.x.T) @ self.y).T
        else:
            n = self.x.shape[1]
            return (((np.linalg.inv(self.x.T @ self.x + (regularization * np.identity(n)))) @ self.x.T) @ self.y).T

    def RMSE(self, Y, _Y):
        return np.sqrt(((Y -_Y) ** 2).mean())

    def GradienteDescedente(self, epochs=5000, learning_rate=0.1):
        errors = []
        initial_weight = np.zeros((1, self.x.shape[1]))
        weight = initial_weight
        for _ in range(epochs):
            prediction = self.regressao_polinomial(self.x, weight)
            error = self.y - prediction
            gradientes = np.mean(error * self.x, axis=0, keepdims=True)
            weight += (learning_rate * gradientes)

            epoch_error = self.RMSE(self.y, prediction)
            errors.append(epoch_error)

        return weight, errors

    def GradienteDescedenteStocastico(self, epochs=5000, learning_rate=0.001):

        errors = []
        initial_weight = np.zeros((1, self.x.shape[1]))
        weight = initial_weight
        for _ in range(epochs):
            shuffle = np.random.permutation(self.x.shape[0])
            x1 = self.x[shuffle]
            y1 = self.y[shuffle]

            for i in range(x1.shape[0]):
                prediction = self.regressao_polinomial(x1[i], weight)
                error = y1[i] - prediction
                gradientes = (x1[i] * error)
                weight = weight + (learning_rate * gradientes)

            prediction = self.regressao_polinomial(self.x, weight)
            errors.append(self.RMSE(self.y, prediction))
        return weight, errors


#%%
