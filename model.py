import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class MLP(nn.Module):

    def __init__(self, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d' % i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d' % i, nn.Linear(
                187 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d' % i, nn.ReLU())

        # self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))
        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 5))
        self.out = nn.Sigmoid()

    def forward(self, x):
        # feature
        # [N, F]

        return self.out(self.mlp(x)).squeeze()