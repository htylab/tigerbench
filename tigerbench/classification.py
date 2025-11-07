import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def _to_labels(y):
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    return y


def accuracy(y_true, y_pred):
    return float(np.mean(_to_labels(y_true) == _to_labels(y_pred)))


def precision(y_true, y_pred, average="macro"):
    return float(precision_score(_to_labels(y_true), _to_labels(y_pred), average=average, zero_division=0))


def recall(y_true, y_pred, average="macro"):
    return float(recall_score(_to_labels(y_true), _to_labels(y_pred), average=average, zero_division=0))


def f1(y_true, y_pred, average="macro"):
    return float(f1_score(_to_labels(y_true), _to_labels(y_pred), average=average, zero_division=0))
