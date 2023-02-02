"""
MNIST classifier training script.

This script uses the torchvision MNIST interface,
and assumes there is an environment variable MNIST_PATH pointing to the data.
"""
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from components import classifiers


num_epochs: int = 1


def train(dataset: DataLoader, model: nn.Module) -> None:
    """Train model.

    Args:
        dataset (DataLoader): batches on which to train.
        model (nn.Module): classification model to train (modified inplace).
    """
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    progbar = tqdm(range(num_epochs))
    num_batches = len(data_train)
    for curr_epoch in progbar:
        for i_batch, (images, labels) in enumerate(dataset):
            # process batch
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update iteration variables
            progbar.set_postfix({
                'batch': f'{i_batch + 1}/{num_batches}',
                'loss': float(loss)
            })


def compute_accuracy(dataset: DataLoader, model: nn.Module) -> float:
    """Compute classification accuracy, percent.

    Args:
        dataset (DataLoader): batches on which to evaluate.
        model (nn.Module): classification model to evaluate.

    Returns:
        float: classification accuracy, percent.
    """
    num_correct, num_seen = 0, 0
    for images, labels in dataset:
        pred = model(images).argmax(dim=-1)
        num_correct += int((pred == labels).sum())
        num_seen += len(images)
    return 100 * num_correct / num_seen


if __name__ == '__main__':
    # load datasets
    data_train = DataLoader(
        batch_size=32,
        dataset=MNIST(
            os.environ['MNIST_PATH'],
            train=True,
            transform=Compose([ToTensor(), Normalize(128, 255)]),
        ),
        shuffle=True,
    )
    data_val = DataLoader(
        batch_size=32,
        dataset=MNIST(
            os.environ['MNIST_PATH'],
            train=False,
            transform=Compose([ToTensor(), Normalize(128, 255)])
        ),
        shuffle=True,
    )

    # initialize model
    my_model = classifiers.AvgPoolClassifier(in_channels=1, num_classes=10)

    # main
    train(data_train, my_model)
    train_acc = compute_accuracy(data_train, my_model)
    val_acc = compute_accuracy(data_val, my_model)
    print(train_acc, val_acc)
