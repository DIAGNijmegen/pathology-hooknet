from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hooknet.model_torch import HookNet
from wholeslidedata.accessories.pytorch.iterator import TorchBatchIterator
from wholeslidedata.iterators import create_batch_iterator


@click.command()
@click.option("--user_confg", type=Path, required=True)
@click.option("--output_folder", type=Path, required=True)
@click.option("--classes", type=int, required=True)
@click.option("--filters", type=int, required=True)
@click.option("--cpus", type=int, required=True)
@click.option("--epochs", type=int, required=True)
@click.option("--steps", type=int, required=True)
def main(user_config, output_folder: Path, classes, filters, cpus, epochs, steps):
    output_folder.mkdir(exists_ok=True, parents=True)

    hooknet = HookNet(classes, n_filters=filters).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(hooknet.parameters(), lr=0.001, momentum=0.9)
    batch_iterators = {
        mode: create_batch_iterator(
            mode=mode,
            user_config=user_config,
            cpus=cpus,
            iterator_class=TorchBatchIterator,
            buffer_dtype="uint8",
        )
        for mode in ["training", "validation"]
    }

    min_valid_loss = np.inf
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch: ", epoch)
        train_loss = 0.0
        hooknet.train()  # Optional when not using Model Specific layer
        for _ in range(steps):
            inputs, labels, info = next(batch_iterators["training"])
            optimizer.zero_grad()
            outputs = hooknet(*inputs)
            loss = None
            for output, label in zip(outputs, labels):
                if loss is None:
                    loss = criterion(output, label)
                else:
                    loss += criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        hooknet.eval()  # Optional when not using Model Specific layer
        for _ in range(steps):
            inputs, labels, _ = next(batch_iterators["validation"])
            outputs = hooknet(*inputs)
            loss = None
            for output, label in zip(outputs, labels):
                if loss is None:
                    loss = criterion(output, label)
                else:
                    loss += criterion(output, label)
            valid_loss += loss.item()

        train_loss /= steps
        valid_loss /= steps
        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}"
        )
        if min_valid_loss > valid_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(hooknet.state_dict(), output_folder / "best_model.pth")

    torch.save(hooknet.state_dict(), output_folder / "last_model.pth")
    print("Finished Training")


if __name__ == "__main__":
    main()
