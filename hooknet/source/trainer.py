from typing import Dict, Type
from random import randint
import sys
import time
import argparse
import random
import os
from abc import abstractmethod, ABC

import numpy as np
import tensorflow as tf

from .model import HookNet

class Trainer(ABC):
    """
    Trainer class

    An abstract class for training and validating a model on batches generated from a batchgenerator. 

    """

    def __init__(self,
                 model: Type,
                 batch_generator: Type,
                 epochs: int,
                 steps: int,
                 batch_size: int) -> None:

        """
        Parameters
        ----------
        model : Model
            Can be any model. set the training and validation methods via the _set_model_functions when subclassing

        batch_generator: BatchGenerator
            Can be any batchgenerator. set the training and validation batch methods via the _set_batch_functions when subclassing

        epochs: int
            The number of epochs the trainer will run

        steps: int
            The number of steps (i.e., batches) in an epoch
        
        batch_size: int
            The number of examples in one batch
            
        """

        self._epochs = epochs
        self._steps = steps

        self._batch_generator = batch_generator
        self._batch_functions = self._set_batch_functions()
        self._model = model
        self._model_functions = self._set_model_functions()

        self._states = ['training', 'validation']

    def train(self):
        """Train loop"""
        for epoch in range(self._epochs):
            epoch_metrics = self._epoch()
            self._update(epoch, epoch_metrics)

    @abstractmethod
    def _update(self, epoch: int, epoch_metrics: Dict) -> None:
        """Abstract method for logging/saving based on computed metrics"""
        pass

    @abstractmethod
    def _set_model_functions(self) -> Dict:
        """Abstract method for setting model functions for each state (e.g., training and validation)"""
        pass

    @abstractmethod
    def _set_batch_functions(self) -> Dict:
        """Abstract method for setting batch functions for each state (e.g., training and validation)"""
        pass

    def _epoch(self) -> Dict:
        """Epoch method which loops over states(training, validation) and returns all computed metrics"""
        epoch_metrics = {state: [] for state in self._states}
        for state in self._states:
            epoch_metrics[state] = [self._step(state) for _ in range(self._steps)]
        return epoch_metrics

    def _step(self, state: str):
        """Step method that retrieves a batch based on the set batch function and applies the model based on the set model function"""
        X_batch, y_batch = self._batch_functions[state]()
        step_metrics = self._model_functions[state](x=X_batch, y=y_batch)
        return step_metrics


class HookNetTrainer(Trainer):

    """ 
    Trainer class specific for the HookNet model. It uses the validation loss to track the best model
    """

    def __init__(self,
                 model: HookNet,
                 batch_generator: Type,
                 epochs: int,
                 steps: int,
                 batch_size: int, 
                 output_path: str) -> None:

        super().__init__(model=model,
                         batch_generator=batch_generator,
                         epochs=epochs,
                         steps=steps,
                         batch_size=batch_size)

        """
        Parameters
        ----------
        model : HookNet
            A HookNet model.

        batch_generator: BatchGenerator
            Can be any batchgenerator with get_training_batch and get_validation_batch methods

        epochs: int
            The number of epochs the trainer will run

        steps: int
            The number of steps (i.e., batches) in an epoch
        
        batch_size: int
            The number of examples in one batch
            
        output_path: int
            The path were the weights are saved
            
        """

        self._best_metric = None
        self._weights_file = os.path.join(output_path, 'weights.h5')
        self._graph = tf.get_default_graph()

    def train(self):
        print('Start training...')
        with self._graph.as_default():
            super().train()

    def _update(self, epoch, epoch_metrics):
        # compute average loss of epoch
        avg_loss = {state: np.array(epoch_metrics[state])[:, 0].mean() for state in self._states}

        # print epoch number and average loss for each state
        print(f'Epoch: {epoch}, {", ".join([f"{state} loss: {avg_loss[state]}" for state in self._states])}')
       
        # check if model improved
        if self._best_metric is None or avg_loss['validation'] < self._best_metric:

            # update best metric
            self._best_metric = avg_loss['validation']

            # save weights
            print(f'Saving weights to: {self._weights_file}')
            self._model.save_weights(self._weights_file)

    def _set_model_functions(self):
        return {'training': self._model.train_on_batch,
                'validation': self._model.test_on_batch}

    def _set_batch_functions(self):
        return {'training': self._batch_generator.get_training_batch,
                'validation': self._batch_generator.get_validation_batch}