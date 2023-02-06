""" Stems for neural networks. """
from torch import nn


class ConvPoolStem(nn.Module):
    """ Very simple 1-stride conv-pool block. """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, inp):
        return self.layers(inp)
