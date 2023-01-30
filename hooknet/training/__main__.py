from pathlib import Path

import click
from hooknet.training.trainer import Trainer


@click.command()
@click.option("--iterator_config", type=Path, required=True)
@click.option("--hooknet_config", type=Path, required=True)
@click.option("--epochs", type=int, required=True)
@click.option("--steps", type=int, required=True)
@click.option("--cpus", type=int, required=True)
@click.option("--project", type=str, required=True)
@click.option("--log_path", type=Path, required=True)
def main(iterator_config, hooknet_config, epochs, steps, cpus, project, log_path):
    trainer = Trainer(
        iterator_config, hooknet_config, epochs, steps, cpus, project, log_path
    )
    trainer.train()


main()
