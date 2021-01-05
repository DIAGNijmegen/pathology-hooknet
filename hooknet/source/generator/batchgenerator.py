from typing import List, Tuple
import numpy as np


class RandomBatchGenerator:

    """
    Random batch generator for illustration and testing purposes
    """

    def __init__(self, batch_size: int, 
                 input_shape: List[int],
                 output_shape : List[int],
                 n_classes: int) -> None:

        self._batch_size = batch_size
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._n_classes = n_classes 

        self.get_training_batch = self._get_batch
        self.get_validation_batch = self._get_batch

    def _get_batch(self) -> Tuple[List, List]:
        return ([np.random.randint(0, 255, (self._batch_size, *self._input_shape))/255.0,
                 np.random.randint(0, 255, (self._batch_size, *self._input_shape))/255.0],
                [np.random.randint(0, self._n_classes, (self._batch_size, self._output_shape[0]*self._output_shape[1], self._n_classes))])
