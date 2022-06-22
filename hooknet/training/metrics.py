from collections import Counter
from copy import deepcopy

import numpy as np
from pycm import ConfusionMatrix


def reshape_evaluation_data(y, predictions, sample_weight, argmax=True):
    n_classes = y.shape[-1]
    y_true = y.reshape(-1, n_classes)
    y_pred = predictions.reshape(-1, n_classes)
    if argmax:
        y_true = np.argmax(y_true, -1)
        y_pred = np.argmax(y_pred, -1)
    if sample_weight is not None:
        sample_weight = sample_weight.reshape(-1)
    return y_true, y_pred, sample_weight


class MetricAccumulater:
    def __init__(self, metric_name, name=None):
        self._metric_name = metric_name
        self._name = self._metric_name if name is None else name
        self._accumalation = []

    def __call__(self):
        metric = {self._name: np.mean(self._accumalation)}
        self.reset()
        return metric

    def update(self, **kwargs):
        self._accumalation.append(kwargs[self._metric_name])

    def reset(self):
        self._accumalation = []


class PredictionMetricAccumulator(MetricAccumulater):
    def __init__(self, evaluation_function, name, argmax=True):
        super().__init__(metric_name=name)
        self._evaluation_function = evaluation_function
        self._argmax = argmax

    def update(self, y, predictions, sample_weight=None, **kwargs):
        y_true, y_pred, sample_weight = reshape_evaluation_data(
            y, predictions, sample_weight, self._argmax
        )
        metric = self._evaluation_function(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        super().update(**{self._name: metric})


class ConfusionMatrixAccumulator:
    def __init__(
        self, metric_names=["F1", "TNR", "TPR", "F1_Macro", "ACC_Macro"], label_map=None
    ):

        self._label_map = label_map
        self._confusion_matrix = {}
        self._metric_names = metric_names
        self.reset()

    def __call__(self):
        confusion_matrix = ConfusionMatrix(matrix=deepcopy(self._confusion_matrix))
        confusion_matrix.relabel(
            mapping={value - 1: key for key, value in self._label_map.items()}
        )

        metrics = {
            metric_name: getattr(confusion_matrix, metric_name)
            for metric_name in self._metric_names
        }
        self.reset()
        return metrics

    def update(self, y, predictions, sample_weight=None, **kwargs):
        n_classes = y.shape[-1]
        actual_vector = np.argmax(y.reshape(-1, n_classes), -1)
        predict_vector = np.argmax(predictions.reshape(-1, n_classes), -1)

        if sample_weight is not None:
            sample_weight = sample_weight.reshape(-1)

        confusion_matrix = ConfusionMatrix(
            actual_vector=actual_vector,
            predict_vector=predict_vector,
            sample_weight=sample_weight,
        )

        for key, value in confusion_matrix.table.items():
            self._confusion_matrix[key].update(value)

    def reset(self):
        self._confusion_matrix = {}
        for label_value in self._label_map.values():
            self._confusion_matrix[label_value - 1] = Counter(
                {
                    inner_label_value - 1: 0
                    for inner_label_value in self._label_map.values()
                }
            )
