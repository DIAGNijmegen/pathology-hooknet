import numpy as np
import torch
from experimart.interoperability.torch.step import TorchStepIterator


def _get_cuda_data(data, label):
    data = np.transpose(data, (1, 0, 4, 2, 3))
    label = np.transpose(label, (1, 0, 2, 3))

    data = torch.tensor(data, device="cuda").float() / 255.0
    label = torch.tensor(label, device="cuda").long()

    return data, label


class HookNetTorchTrainingStepIterator(TorchStepIterator):
    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label, *_ = next(self._data_iterator)
            data_cuda, label_cuda = _get_cuda_data(data, label)
            self._components.optimizer.zero_grad()
            output = self._model(*data_cuda)["out"]
            loss = self._components.criterion(output, label_cuda[0])
            loss.backward()
            self._components.optimizer.step()
            metrics = self.get_metrics(label_cuda[0], output)
            metrics["learning_rate"] = self._components.scheduler.get_last_lr()[0]
            yield {"loss": loss.item(), **metrics}
        self._components.scheduler.step()


class HookNetTorchValidationStepIterator(TorchStepIterator):
    def steps(self):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label, *_ = next(self._data_iterator)
                data_cuda, label_cuda = _get_cuda_data(data, label)

                output = self._model(*data_cuda)
                output = output["out"]
                loss = self._components.criterion(output, label_cuda[0])
                metrics = self.get_metrics(label_cuda[0], output)
                yield {"loss": loss.item(), **metrics}
