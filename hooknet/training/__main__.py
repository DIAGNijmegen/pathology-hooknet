import glob
import os
from pathlib import Path

import click
from creationism.utils import open_yaml
from hooknet.configuration.config import create_hooknet
from hooknet.training.metrics import (ConfusionMatrixAccumulator,
                                      MetricAccumulater,
                                      PredictionMetricAccumulator)
from hooknet.training.tracker import WandbTracker
from sklearn.metrics import log_loss
from wholeslidedata.iterators import create_batch_iterator
from hooknet.training.train import train, MODES


@click.command()
@click.option("--iterator_config", type=Path, required=True)
@click.option("--hooknet_config", type=Path, required=True)
@click.option("--epochs", type=int, required=True)
@click.option("--steps", type=int, required=True)
@click.option("--cpus", type=int, required=True)
@click.option("--project", type=str, required=True)
@click.option("--log_path", type=Path, required=True)
def main(iterator_config, hooknet_config, epochs, steps, cpus, project, log_path):
    tracker = WandbTracker(project=project, log_path=log_path)

    tracker.save(str(iterator_config))
    tracker.save(str(hooknet_config))
        
    iterators = {
        mode: create_batch_iterator(mode=mode, user_config=iterator_config, cpus=cpus)
        for mode in MODES
    }

    hooknet = create_hooknet(hooknet_config, mode='training')
    model_functions = {
        "training": hooknet.train_on_batch,
        "validation": hooknet.predict_on_batch,
    }
    weights_file = "./hooknet_weights.h5"


    label_map = open_yaml(iterator_config)['hooknet']['default']['label_map']

    metrics = {
        "training": [
            MetricAccumulater("loss", "Loss+L2"),
            MetricAccumulater("accuracy"),
            MetricAccumulater("categorical_crossentropy", "Loss"),
        ],
        "validation": [
            ConfusionMatrixAccumulator(
                metric_names=("F1", "F1_Macro", "ACC_Macro"), label_map=label_map
            ),
            PredictionMetricAccumulator(log_loss, "Loss", argmax=False)
        ],
    }

    print("Training...")
    try:
        train(
            hooknet=hooknet,
            epochs=epochs,
            steps=steps,
            iterators=iterators,
            model_functions=model_functions,
            metrics=metrics,
            tracker=tracker,
            weights_file=weights_file,
        )
    except Exception as exception:
        # finish tracker
        print("save wandb files...")
        for file in glob.glob(os.path.join(log_path, "*.log")):
            tracker.save(file)
        raise exception

main()
