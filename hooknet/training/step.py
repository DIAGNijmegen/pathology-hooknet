from experimart.interoperability.torch.step import TorchStepIterator
import torch
import numpy as np

def _get_cuda_data(data, label):
    data = np.array(data)
    label = np.array(label)
    data = np.transpose(data, (1,0,4,2,3)).astype('float32')
    label = np.transpose(label, (1,0,2,3)).astype('int16')
    return torch.from_numpy(data).cuda(), torch.from_numpy(label).cuda().long()


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
            yield {"loss": loss.item(), **metrics}
        self._components.scheduler.step()


class HookNetTorchValidationStepIterator(TorchStepIterator):
    def steps(self):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label, *_ = next(self._data_iterator)
                data_cuda,  label_cuda  = _get_cuda_data(data, label)
                
                output = self._model(*data_cuda)
                output = output["out"]
                loss = self._components.criterion(output, label_cuda[0])
                metrics = self.get_metrics(label_cuda[0], output)
                yield {"loss": loss.item(), **metrics}
