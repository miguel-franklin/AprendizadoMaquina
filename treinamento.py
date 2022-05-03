import numpy as np
from Models.model import Model

def LinearRegression(Model):

    def __init__(self, w, x):
        self.w = w
        self.x = x

    def predict(self):
        return self.w @ self.x


def GD(model:Model, epochs=1000):
    """
    Gradiente Descedente
    :param model:
    :param epochs:
    :return:
    """
    errors = []
    initial_weight = np.zeros((1, self.x.shape[1]))
    weight = initial_weight
    for _ in range(epochs):
        prediction = model.predict(self.x, weight)
        error = self.y - prediction
        gradientes = np.mean(error * self.x, axis=0, keepdims=True)
        weight += (learning_rate * gradientes)

        epoch_error = self.RMSE(self.y, prediction)
        errors.append(epoch_error)

    return weight, errors
