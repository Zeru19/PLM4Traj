from torch import nn


class Loss(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = f'LOSS_{name}'
        