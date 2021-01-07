import numpy as np
from experiment.callbacks import ExperimentCallback


class HookNetReshape(ExperimentCallback):
    def __init__(self, multi_loss=False):
        self._multi_loss = multi_loss

    def __call__(self, x_batch: np.ndarray, y_batch: np.ndarray):
        x_list_batch = [[], []]
        y_list_batch = [[], []]

        for batch_sample in x_batch:
            batch_sample = dict(sorted(batch_sample.items()))
            for idx, (key, value) in enumerate(batch_sample.items()):
                x_list_batch[idx].append(value)

        for batch_sample in y_batch:
            batch_sample = dict(sorted(batch_sample.items()))
            for idx, (key, value) in enumerate(batch_sample.items()):
                y_list_batch[idx].append(value)

        x_list_batch[0] = np.array(x_list_batch[0])
        x_list_batch[1] = np.array(x_list_batch[1])
        y_list_batch[0] = np.array(y_list_batch[0])
        y_list_batch[1] = np.array(y_list_batch[1])
        if self._multi_loss:
            return x_list_batch, y_list_batch

        return x_list_batch, y_list_batch[0]