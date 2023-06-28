"""
@file:nets
@author:qzz
@date:2023/3/3
@encoding:utf-8
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


class PolicyNet(nn.Module):
    def __init__(self):
        """
        A policy net to output policy distribution.
        """
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 38)
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        policy = F.log_softmax(out, -1)
        return policy


class PolicyNet2(nn.Module):
    def __init__(self):
        """
        A policy net to output policy distribution.
        """
        super(PolicyNet2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 38)
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        policy = F.log_softmax(out, -1)
        return policy


class PolicyNetRelu(nn.Module):
    def __init__(self, num_hidden_layers: int = 4, hidden_dim: int = 1024, input_dim: int = 480,
                 output_dim: int = 38):
        """
        A policy net to output policy distribution.
        """
        super(PolicyNetRelu, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        ff_layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for i in range(1, self.num_hidden_layers):
            ff_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            ff_layers.append(nn.ReLU())
        ff_layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.net = nn.Sequential(*ff_layers)

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        policy = F.log_softmax(out, -1)
        return policy


class ValueNet(nn.Module):

    def __init__(self):
        """A value net as critic."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        )

    def forward(self, state: torch.Tensor):
        value = self.net(state)
        return value


class PerfectValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(636, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1),
            nn.Tanh()
        )

    def forward(self, p_state: torch.Tensor):
        value = self.net(p_state)
        return value


class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(229, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 4)
        )

    def forward(self, p_state: torch.Tensor):
        value = self.net(p_state)
        return value


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(518, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
        )
        self.fc_q = nn.Linear(1024, 38)

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        q = self.fc_q(h)
        return q


class DoubleDummyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(208, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 20)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
