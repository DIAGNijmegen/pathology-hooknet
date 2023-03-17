from experimart.interoperability.torch.step import TorchStepIterator
import torch

class HookNetTorchTrainingStepIterator(TorchStepIterator):
    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label, *_ = next(self._data_iterator)
            self._components.optimizer.zero_grad()
            output = self._model(*data)["out"]
            loss = self._components.criterion(output, label[0])
            loss.backward()
            self._components.optimizer.step()
            metrics = self.get_metrics(label[0], output)
            yield {"loss": loss.item(), **metrics}
        self._components.scheduler.step()


class HookNetTorchValidationStepIterator(TorchStepIterator):
    def steps(self):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label, *_ = next(self._data_iterator)
                output = self._model(*data)
                output = output["out"]
                loss = self._components.criterion(output, label[0])
                metrics = self.get_metrics(label[0], output)
                yield {"loss": loss.item(), **metrics}
