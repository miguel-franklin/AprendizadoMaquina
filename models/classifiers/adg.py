import numpy as np

from models import model
from models.metrics.metrics_factory import get_classification_metrics

class ADG(model.Model):

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

    def matrix_cov(self, x, y):
        cov_matrices = np.array([np.cov(x[y == 0], rowvar=False)])
        for class_k in range(1, np.max(y) + 1):
            covariance_matrix = np.cov(x[y == class_k], rowvar=False)
            cov_matrices = np.concatenate([cov_matrices, [covariance_matrix]])

        return cov_matrices

    def matrix_means(self, x, y):
        means = np.array(np.mean(x[y == 0], axis=0, keepdims=True))
        for class_k in range(1, np.max(y) + 1):
            mean_k = np.mean(x[y == class_k], axis=0, keepdims=True)
            means = np.concatenate([means, mean_k])
        return means

    def fit(self, x, y, training=None):
        self.k = len(np.unique(y))

        self.k_prop = self.k_prop_prob(y)
        self.variances = self.matrix_cov(x, y)
        self.means = self.matrix_means(x, y)

    def predict(self, x):
        _y = []

        for i in range(x.shape[0]):
            probs = []

            for k in range(self.k):
                k_log = np.log(self.k_prop[k])

                det_var = -0.5 * np.log(np.linalg.det(self.variances[k]))

                std = x[i] - self.means[k]
                inv_var = (-0.5) * (std @ (np.linalg.pinv(self.variances[k]) @ std.T))

                probs.append(k_log+det_var+inv_var)

            _y.append(np.argmax(probs))

        return np.array(_y)

    def predict_train(self, x, weights):
        pass

