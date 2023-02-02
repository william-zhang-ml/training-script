"""
MNIST classifier training script.

This script uses the torchvision MNIST interface,
and assumes there is an environment variable MNIST_PATH pointing to the data.
"""
import argparse
import json
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from components import classifiers


# define default config
CONFIG = {
    'out_dir':    None,
    'run_name':   None,
    'cls_class':  None,
    'cls_kwargs': None,
    'batch_size': 32,
    'num_epochs': 1
}
TYPES = {
    'out_dir':    str,
    'run_name':   str,
    'cls_class':  str,
    'cls_kwargs': json.loads,
    'batch_size': int,
    'num_epochs': int
}
HELP = {
    'out_dir':    '(str) directory in which to save outputs',
    'run_name':   '(str) identifier used to tell current run from others',
    'cls_class':  '(str) classifier class name',
    'cls_kwargs': '(dict) classifier initialize args',
    'batch_size': '(int) number of training images per gradient step',
    'num_epochs': '(int) number of training epochs'
}


def train(dataset: DataLoader,
          model: nn.Module,
          board: SummaryWriter = None) -> None:
    """Train model.

    Args:
        dataset (DataLoader): batches on which to train.
        model (nn.Module): classification model to train (modified inplace).
        board (SummaryWriter): torch tensorboard instance
    """
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    progbar = tqdm(range(CONFIG['num_epochs']))
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
    # overwrite default config values w/CLI inputs
    parser = argparse.ArgumentParser()
    for arg_name, default_val in CONFIG.items():
        parser.add_argument(
            arg_name if default_val is None else f'--{arg_name}',
            default=default_val,
            help=HELP[arg_name],
            type=TYPES[arg_name]
        )
    CONFIG.update(vars(parser.parse_args()))

    # setup output subdirectories
    run_dir = os.path.join(CONFIG['out_dir'], 'runs', CONFIG['run_name'])
    tb_dir = os.path.join(CONFIG['out_dir'], '_tensorboard')
    if os.path.isdir(run_dir):
        raise FileExistsError('found preexisting run at desired location.')
    os.makedirs(run_dir)
    my_board = SummaryWriter(tb_dir)

    # write config to JSON
    file_name = os.path.join(run_dir, 'config.json')
    with open(file_name, 'w', encoding='UTF-8') as file:
        json.dump(CONFIG, file, indent=4)

    # load datasets
    data_train = DataLoader(
        batch_size=CONFIG['batch_size'],
        dataset=MNIST(
            os.environ['MNIST_PATH'],
            train=True,
            transform=Compose([ToTensor(), Normalize(128, 255)]),
        ),
        shuffle=True,
    )
    data_val = DataLoader(
        batch_size=CONFIG['batch_size'],
        dataset=MNIST(
            os.environ['MNIST_PATH'],
            train=False,
            transform=Compose([ToTensor(), Normalize(128, 255)])
        ),
        shuffle=True,
    )

    # initialize model
    my_model = getattr(classifiers, CONFIG['cls_class'])(
        **CONFIG['cls_kwargs'])  # pylint: disable=not-a-mapping

    # main
    train(data_train, my_model)
    acc_train = compute_accuracy(data_train, my_model)
    acc_val = compute_accuracy(data_val, my_model)
    hparams = {
        'batch_size': CONFIG['batch_size'],
        'num_epochs': CONFIG['num_epochs'],
        'cls': CONFIG['cls_class']
    }
    hparams.update(CONFIG['cls_kwargs'])
    my_board.add_hparams(
        hparams,
        {
            'ACC/train': acc_train,
            'ACC/val': acc_val
        },
        run_name=CONFIG['run_name']
    )
