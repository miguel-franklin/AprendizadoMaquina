import numpy as np

from models import model
from models.metrics.metrics_factory import get_classification_metrics


class NaiveGauss(model.Model):

    def __init__(self, normalization=None, metrics='F1'):
        self.weights = None
        self.normalization = normalization
        self.metrics = get_classification_metrics(metrics)

    def initial_weights(self, x):
        pass

    def k_prop_prob(self, y):
        """
        Calculate probabilites of classes
        :param y:
        :return:
        """
        k, k_count = np.unique(y, return_counts=True)
        return k_count / y.shape[0]

    def matrix_var(self, x, y):
        att = lambda k: x[y == k]
        variances = np.array([(att(0)).var(axis=0, ddof=1)])
        for class_k in range(1, np.max(y) + 1):
            variance = att(class_k).var(axis=0, ddof=1)
            variances = np.concatenate([variances, [variance]])
        return variances

    def matrix_means(self, x, y):
        att = lambda k: x[y == k]
        means = np.array(np.mean(att(0), axis=0, keepdims=True))
        for class_k in range(1, np.max(y) + 1):
            mean_k = np.mean(att(class_k), axis=0, keepdims=True)
            means = np.concatenate([means, mean_k])
        return means

    def fit(self, x, y, training=None):
        self.k = len(np.unique(y))

        self.k_prop = self.k_prop_prob(y)
        self.variances = self.matrix_var(x, y)
        self.means = self.matrix_means(x, y)

    def predict(self, x):
        _y = []

        for i in range(x.shape[0]):
            probs = []

            for k in range(self.k):
                k_log = np.log(self.k_prop[k])

                k_log_var = 0
                for d in range(x.shape[1]):
                    k_log_var += np.log(2 * np.pi * self.variances[k][d])
                k_log_var *= -0.5

                k_means_var = 0
                for d in range(x.shape[1]):
                    k_means_var += ((x[i][d] - self.means[k][d]) ** 2) / self.variances[k][d]
                k_means_var *= -0.5

                probs.append(k_log + k_log_var + k_means_var)

            _y.append(np.argmax(probs))

        return np.array(_y)

    def predict_train(self, x, weights):
        pass
