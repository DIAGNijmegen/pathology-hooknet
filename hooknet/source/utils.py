import numpy as np
from experiment.callbacks import TrainCallback

def clean_weights(masks):
    return np.clip(np.sum(masks, axis=-1), 0, 1)

class HookNetReshape(TrainCallback):
    def __init__(self, multi_loss=False):
        self._multi_loss = multi_loss

    def __call__(self, x_batch: np.ndarray, y_batch: np.ndarray, sample_weight):
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
            return (
                x_list_batch,
                y_list_batch,
                [clean_weights(y_list_batch[0]), clean_weights(y_list_batch[1])],
            )

        return x_list_batch, y_list_batch[0], clean_weights(y_list_batch[0])