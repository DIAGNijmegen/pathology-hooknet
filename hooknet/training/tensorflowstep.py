from experimart.interoperability.tensorflow.step import TensorflowStepIterator


def _transpose_data(x_batch, y_batch):
    x_batch = list(x_batch.transpose(1, 0, 2, 3, 4))
    y_batch = y_batch.transpose(1, 0, 2, 3, 4)[0]
    return x_batch, y_batch


class HookNetTensorflowTrainingStepIterator(TensorflowStepIterator):
    def steps(self):
        for _ in range(len(self)):
            x_batch, y_batch, _ = next(self._data_iterator)
            x_batch, y_batch = _transpose_data(x_batch, y_batch)

            yield self._model.train_on_batch(
                x=x_batch, y=y_batch, return_dict=True, metrics=self._metrics
            )


class HookNetTensorflowValidationStepIterator(TensorflowStepIterator):
    def steps(self):
        lr = self._update_learning_rate()
        for _ in range(len(self)):
            metrics = {} if lr is None else {"learning_rate": lr}
            x_batch, y_batch, _ = next(self._data_iterator)
            x_batch, y_batch = _transpose_data(x_batch, y_batch)
            predictions = self._model.predict_on_batch(x=x_batch, argmax=False)
            metrics.update(self.get_metrics(y_batch, predictions))
            yield metrics
