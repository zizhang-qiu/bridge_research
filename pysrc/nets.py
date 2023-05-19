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
            nn.Linear(571, 2048),
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
            nn.Linear(1024, 1),
            nn.Tanh()
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
            nn.Linear(1024, 1)
        )

    def forward(self, p_state: torch.Tensor):
        value = self.net(p_state)
        return value
