import numpy as np

from .abstract import AbstractMeasure


class Recall(AbstractMeasure):
    def measure(self, y, _y):
        _y = np.round(_y)

        true_positives = np.sum(np.logical_and((_y == 1), (y == 1)))
        false_negatives = np.sum(np.logical_and((_y == 0), (y == 1)))

        p = true_positives + false_negatives
        if p > 1.0e-7:
            return true_positives / p
        return 0.0


class Precision(AbstractMeasure):
    def measure(self, y, _y):
        _y = np.round(_y)

        true_positives = np.sum(np.logical_and((_y == 1), (y == 1)))
        false_positives = np.sum(np.logical_and((_y == 1), (y == 0)))

        pp = true_positives + false_positives
        if pp > 1.0e-7:
            return true_positives / pp
        return 0.0


class F1Score(AbstractMeasure):
    def measure(self, y, _y):
        _y = np.round(_y)

        precision = Precision().measure(y, _y)
        recall = Recall().measure(y, _y)

        if precision + recall > 1.0e-7:
            return 2 * ((precision * recall) / (precision + recall))
        return 0.0


class BinaryAccuracy(AbstractMeasure):
    def measure(self, y, _y):
        _y = np.round(_y)
        return np.mean(np.round(y) == np.round(_y))
