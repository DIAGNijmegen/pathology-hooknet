from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hooknet.model_torch import HookNet
from wholeslidedata.interoperability.pytorch.iterator import TorchBatchIterator
from wholeslidedata.iterators import create_batch_iterator
from hooknet.training.tracker import WandbTracker
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import time




class Trainer:
    def __init__(
        self, iterator_config, classes, filters, epochs, steps, cpus, project, log_path
    ):
        self._iterator_config = iterator_config
        self._cpus = cpus
        self._classes= classes
        self._filters = filters
        self._project = project
        self._log_path = Path(log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._epochs = epochs
        self._steps = steps

    def train(self):

        tracker = WandbTracker(project=self._project, log_path=self._log_path)
        tracker.save(str(self._iterator_config))

        hooknet = HookNet(self._classes, n_filters=self._filters).cuda()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(hooknet.parameters(), lr=0.05)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        batch_iterators = {
            mode: create_batch_iterator(
                mode=mode,
                user_config=self._iterator_config,
                cpus=self._cpus,
                iterator_class=TorchBatchIterator,
                buffer_dtype="uint8",
            )
            for mode in ["training"]
        }

        min_valid_loss = np.inf
        for epoch in range(self._epochs):  # loop over the dataset multiple times
            print("Epoch: ", epoch)
            train_loss = 0.0
            hooknet.train()  # Optional when not using Model Specific layer
            print('training')
            for _ in tqdm(range(self._steps)):
                inputs, labels, info = next(batch_iterators["training"])
                optimizer.zero_grad()
                output = hooknet(*inputs)
                loss = criterion(output, labels[0].long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            valid_loss = 0.0
#             with torch.no_grad():
#                 hooknet.eval()  # Optional when not using Model Specific layer
#                 print('validation')
#                 for _ in tqdm(range(self._steps)):
#                     inputs, labels, _ = next(batch_iterators["validation"])
#                     outputs = hooknet(*inputs)
#                     loss = criterion(outputs[0], labels[0].long())
#                     valid_loss += loss.item()

            train_loss /= self._steps
#             valid_loss /= self._steps

            tracker.update({'train_loss': train_loss, 'valid_loss': valid_loss})
            print(
                f"Epoch {epoch+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}"
            )
#             if min_valid_loss > valid_loss:
#                 print(
#                     f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
#                 )
#                 min_valid_loss = valid_loss
#                 # Saving State Dict
#                 torch.save(hooknet.state_dict(), self._log_path / "best_model.pth")

            torch.save(hooknet.state_dict(), self._log_path / "last_model.pth")
            print("Finished Training")



@click.command()
@click.option("--iterator_config", type=Path, required=True)
@click.option("--output_folder", type=Path, required=True)
@click.option("--classes", type=int, required=True)
@click.option("--filters", type=int, required=True)
@click.option("--cpus", type=int, required=True)
@click.option("--epochs", type=int, required=True)
@click.option("--steps", type=int, required=True)
@click.option("--project", type=str, required=True)
@click.option("--log_path", type=Path, required=True)
def main(iterator_config, output_folder: Path, classes, filters, cpus, epochs, steps):
    pass


if __name__ == "__main__":
    main()
