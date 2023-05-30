import numpy as np
from experimart.interoperability.torch.io import convert_data_to_device
from experimart.interoperability.torch.step import (
    TorchStepIterator,
    TorchMultiInputStepIterator,
    TorchTrainingStepIterator,
    TorchValidationStepIterator,
)


class HookNetTorchStepIterator(TorchMultiInputStepIterator, TorchStepIterator):
    def _get_data(self):
        data, label, *_ = next(self._data_iterator)
        data = np.transpose(data, (1, 0, 4, 2, 3)) # inputs first
        label = np.transpose(label, (1, 0, 2, 3))[0] # inputs first and high-res label only
        return convert_data_to_device(data, label, device=self._device)


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
