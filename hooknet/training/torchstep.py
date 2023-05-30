import numpy as np
import torch
from experimart.interoperability.torch.step import (
    TorchStepIterator,
    TorchTrainingStepIterator,
    TorchValidationStepIterator,
)


class HookNetTorchStepIterator(TorchStepIterator):
    def _get_data(self):
        data, label, *_ = next(self._data_iterator)
        data = np.transpose(data, (1, 0, 4, 2, 3))
        label = np.transpose(label, (1, 0, 2, 3))[0]

        data = torch.tensor(data, device="cuda").float() / 255.0
        label = torch.tensor(label, device="cuda").long()
        return data, label

    def _get_output(self, data):
        return self._model(*data)["out"]


class HookNetTorchTrainingStepIterator(
    HookNetTorchStepIterator,
    TorchTrainingStepIterator,
):
    ...


class HookNetTorchValidationStepIterator(
    HookNetTorchStepIterator,
    TorchValidationStepIterator,
):
    ...
