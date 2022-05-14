import numpy as np
from models.model import Model


def LinearRegression(Model):
    def __init__(self, w, x):
        self.w = w
        self.x = x

    def predict(self):
        return self.w @ self.x


def RMSE(Y, _Y):
    return np.sqrt(((Y - _Y) ** 2).mean())


def GD(model: Model, x, y, epochs=1000, learning_rate=0.1):
    """

    :param model:
    :param x:
    :param y:
    :param epochs:
    :param learning_rate:
    :return:
    """
    errors = []
    for _ in range(epochs):
        prediction = model.predict_train(x, model.weights)
        error = y - prediction
        gradientes = np.mean(error * x, axis=0, keepdims=True)
        model.weights += (learning_rate * gradientes)

        epoch_error = model.metrics.measure(y, prediction)
        errors.append(epoch_error)

    return model.weights, errors


def SGD(model: Model, x, y, epochs=500, learning_rate=0.001):
    """
    :param model:
    :param x:
    :param y:
    :param epochs:
    :param learning_rate:
    :return:
    """

    errors = []
    for _ in range(epochs):
        shuffle = np.random.permutation(x.shape[0])
        x1 = x[shuffle]
        y1 = y[shuffle]
        for i in range(x1.shape[0]):
            prediction = model.predict_train(x1[i], model.weights)
            error = y1[i] - prediction
            gradientes = (x1[i] * error)
            model.weights = model.weights + (learning_rate * gradientes)

        prediction = model.predict_train(x, model.weights)
        errors.append(RMSE(y, prediction))
    return model.weights, errors
