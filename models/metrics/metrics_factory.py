from models.metrics.abstract import AbstractMeasure
from models.metrics.classification import F1Score, Recall, Precision, BinaryAccuracy
import numpy as np

def get_classification_metrics(metrics) -> AbstractMeasure:
    if metrics == 'F1':
        return F1Score()
    elif metrics == 'Recall':
        return Recall()
    elif metrics == 'Precision':
        return Precision()
    elif metrics == 'Accuracy':
        return BinaryAccuracy()
    else:
        ValueError("metrics doesn't exist")


def measure_all(y, _y):
    measurers = {
        'F1': F1Score(),
        'Accuracy': BinaryAccuracy(),
        'Precision': Precision(),
        'Recall': Recall()
    }
    results = {}
    for k, v in measurers.items():
        results[k] = v.measure(y, _y)
    return results

def print_result(modelo, results):
    f1 = []
    acc = []
    prec = []
    recall = []
    for item in results:
        f1.append(item['F1'])
        acc.append(item['Accuracy'])
        prec.append(item['Precision'])
        recall.append(item['Recall'])
    print("Modelo: {}, F1 Score: {}".format(modelo, np.sum(f1) / len(f1)))
    print("Modelo: {}, Accuracy: {}".format(modelo, np.sum(acc) / len(acc)))
    print("Modelo: {}, Precision: {}".format(modelo, np.sum(prec) / len(prec)))
    print("Modelo: {}, Recall: {}".format(modelo, np.sum(recall) / len(recall)))

