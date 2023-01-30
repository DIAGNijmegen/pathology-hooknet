import glob
import os
from pathlib import Path

from dicfg.reader import _open_yaml_config
from hooknet.model import create_hooknet
from hooknet.training.metrics import (
    ConfusionMatrixAccumulator,
    MetricAccumulater,
    PredictionMetricAccumulator,
)
from hooknet.training.tracker import WandbTracker
from sklearn.metrics import log_loss
from tqdm import tqdm
from wholeslidedata.iterators import create_batch_iterator
import numpy as np

MODES = ["training", "validation"]


class KerasUpdateLearningRate:
    def __init__(
        self,
        model,
        decay_rate=0.5,
        decay_steps=[2, 4, 1000, 5000, 10_000, 50_000, 100_000],
    ):
        self._model = model
        self._lr = self._model.optimizer.lr.numpy()
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._index = 0

    def __call__(self):
        if self._index in self._decay_steps:
            self._lr *= self._decay_rate
            self._model.optimizer.lr.assign(self._lr)
            print(f"updated learning rate={self._lr}")
        self._index += 1





class Trainer:
    def __init__(
        self, iterator_config, hooknet_config, epochs, steps, cpus, project, log_path
    ):
        self._iterator_config = iterator_config
        self._hooknet_config = hooknet_config
        self._cpus = cpus
        self._project = project
        self._log_path = Path(log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._epochs = epochs
        self._steps = steps
        self._weights_file = self._log_path / "hooknet_weights.h5"

    def train(self):

        tracker = WandbTracker(project=self._project, log_path=self._log_path)
        tracker.save(str(self._iterator_config))
        tracker.save(str(self._hooknet_config))

        iterators = {
            mode: create_batch_iterator(
                mode=mode, user_config=self._iterator_config, cpus=self._cpus, buffer_dtype=np.uint8
            )
            for mode in MODES
        }

        hooknet = create_hooknet(self._hooknet_config)

        # label_map = self._iterators['training'].dataset.labels.map

        metrics = {
            "training": [
                MetricAccumulater("loss", "Loss+L2"),
                MetricAccumulater("accuracy"),
                MetricAccumulater("categorical_crossentropy", "Loss"),
            ],
            "validation": [
                # ConfusionMatrixAccumulator(
                #     metric_names=("F1", "F1_Macro", "ACC_Macro"), label_map=label_map
                # ),
                PredictionMetricAccumulator(log_loss, "Loss", argmax=False),
            ],
        }

        best_metric = None
        update_learning_rate = KerasUpdateLearningRate(hooknet)
        print("training labels", iterators["training"].dataset.labels.names)
        print("validation labels", iterators["validation"].dataset.labels.names)
        for epoch in range(self._epochs):
            for mode in MODES:
                for _ in range(self._steps):
                    x_batch, y_batch, _ = next(iterators[mode])
                    x_batch = list(x_batch.transpose(1, 0, 2, 3, 4))
                    y_batch = y_batch.transpose(1, 0, 2, 3, 4)[0]

                    if mode == "training":
                        out = hooknet.train_on_batch(
                            x=x_batch, y=y_batch, return_dict=True
                        )
                    else:
                        predictions = hooknet.predict_on_batch(
                            x=x_batch, argmax=False
                        )
                        out = {"predictions": predictions, "y": y_batch}

                    for metric in metrics[mode]:
                        metric.update(**out)

                    update_learning_rate()

                metrics_data = {}
                for metric in metrics[mode]:
                    metrics_data.update(metric())

                mode_metrics = {
                    mode + "_" + name: value for name, value in metrics_data.items()
                }
                print(epoch, mode_metrics)
                tracker.update(mode_metrics)

                if mode == "validation":
                    # check if model improved
                    if best_metric is None or metrics_data["Loss"] < best_metric:
                        # update best metric
                        best_metric = metrics_data["Loss"]
                        print("new best metric: ", best_metric)
                        # tracker.update_best(best_metric)
                        # save weights
                        print(f"Saving weights to: {self._weights_file}")
                        hooknet.save_weights(self._weights_file)

        hooknet.save_weights(self._log_path / "hooknet_last_model.h5")

        print("save wandb files...")
        for file in glob.glob(os.path.join(self._log_path, "*.log")):
            tracker.save(file)
        for mode in MODES:
            iterators[mode].stop()
        
