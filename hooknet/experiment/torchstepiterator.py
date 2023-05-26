from experiment.training.step import StepIterator
import torch

class TorchTrainingStepIterator(StepIterator):
    def __init__(
        self, model, data_iterator, num_steps, metrics, optimizer, criterion, scheduler
    ):
        super().__init__(model, data_iterator, num_steps, metrics)
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler

    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label, info = next(self._data_iterator)
            self._optimizer.zero_grad()
            output = self._model(*data)["out"]
            loss = self._criterion(output[0], label[0])
            loss.backward()
            self._optimizer.step()
            metrics = self.get_metrics(label[0], output[0])
            yield {"loss": loss.item(), **metrics}
        self._scheduler.step()


class TorchValidationStepIterator(StepIterator):
    def __init__(self, model, data_iterator, num_steps, metrics, criterion):
        super().__init__(model, data_iterator, num_steps, metrics)
        self._criterion = criterion

    def steps(self, data, label):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label, info = next(self._data_iterator)
                output = self._model(*data)
                output = output["out"]
                loss = self._criterion(output[0], label[0])
                metrics = self.get_metrics(label[0], output[0])
                yield {"loss": loss.item(), **metrics}
