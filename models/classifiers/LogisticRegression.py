import numpy as np
from models import model
from models.metrics.metrics_factory import get_classification_metrics
from models.normalization import MinMaxScaler


class LogisticRegression(model.Model):

    def __init__(self, normalization=None, metrics='F1'):
        self.weights = None
        self.normalization = normalization
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.metrics = get_classification_metrics(metrics)

    def initial_weights(self, x):
        return np.zeros((1, x.shape[1]))

    def fit(self, x, y, training=None):
        y = y.reshape(y.shape[0], 1)

        x = self.x_scaler.fit(x).transform(x)
        y = self.y_scaler.fit(y).transform(y)

        ones = np.ones(x.shape[0])
        x = np.column_stack((ones, x[:, ]))

        self.weights = self.initial_weights(x)

        return training(self, x, y)

    def __sigmoid__(self, x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def predict_train(self, x, weights):
        return self.__sigmoid__(x @ weights.T)

    def predict(self, x, weights):
        x = self.x_scaler.transform(x)
        x = np.c_[np.ones(x.shape[0]), x]

        y = np.round(self.predict_train(x, self.weights))
        return self.y_scaler.inverse_transform(y)
