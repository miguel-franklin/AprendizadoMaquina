from models.metrics.abstract import AbstractMeasure
from models.metrics.classification import F1Score, Recall, Precision, BinaryAccuracy
import matplotlib.pyplot as plt
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


def add_result(k, values):
    return {'k': k, 'values': values}


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
    print("Modelo: {}, F1 Score: Mean: {} - Std: {}".format(modelo, np.mean(f1), np.std(f1)))
    print("Modelo: {}, Accuracy: Mean: {} - Std: {}".format(modelo, np.mean(acc), np.std(acc)))
    print("Modelo: {}, Precision: Mean: {} - Std: {}".format(modelo, np.mean(prec), np.std(prec)))
    print("Modelo: {}, Recall: Mean: {} - Std: {}".format(modelo, np.mean(recall), np.std(recall)))


def plot_results(figure_title, results):
    fig, axs = plt.subplots(1, len(results), figsize=(20, 8))

    plot_info = []
    for i in range(len(results)):
        plot_info.append((results[i]['k'], axs[i], results[i]['values'], None))

    x, width = np.arange(4), 0.9
    for key, ax, values, y_limit in plot_info:
        items = [
            [i['F1'] for i in values],
            [i['Accuracy'] for i in values],
            [i['Precision'] for i in values],
            [i['Recall'] for i in values],
        ]

        mape_cv_mean = [np.round(np.mean(np.abs(item)), 2) for item in items]
        mape_cv_std = [np.round(np.std(item), 2) for item in items]

        bar1 = ax.bar(
            x=x,
            height=mape_cv_mean,
            width=width,
            yerr=mape_cv_std,
            color=["C0", "C1", "C2", "C3"],
        )
        ax.bar_label(bar1, label_type='center', fontsize=14)
        ax.bar_label(bar1, labels=['±%.2f' % e for e in mape_cv_std], fontsize=14)

        ax.set(
            xlabel=key,
            title='Score',
            xticks=x,
            xticklabels=["F1", "Acc", "Pre", "Rec"],
            ylim=None,
        )
    fig.suptitle(figure_title)

