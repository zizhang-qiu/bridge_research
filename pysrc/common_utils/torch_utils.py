#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:torch_utils.py
@time:2023/02/17
"""
from typing import List, Optional, Sequence

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn


def print_parameters(model: nn.Module):
    """
    Print a network's parameter

    Args:
        model (nn.Module): the model to print

    Returns:
        No returns
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def get_device(model: nn.Module):
    """
    Returns the device on which a PyTorch model is located.

    Args:
        model: A PyTorch model.

    Returns:
        A string representing the device on which the model is located,
        e.g. 'cpu' or 'cuda:0'.
    """
    return str(next(model.parameters()).device)


def reset_network_params(model: nn.Module, value: float):
    """
    Set the network's parameter to the value
    Args:
        model: The network module
        value: The value to set

    Returns:
        No returns.
    """
    for param in model.parameters():
        param.data.fill_(value)


def create_shared_dict(net: nn.Module):
    """
    Create a shared dict from a network. The shared dict can be used in multiprocess context.
    Args:
        net: The network module

    Returns:
        The shared dict.
    """
    net_state_dict = net.state_dict()
    for k, v in net_state_dict.items():
        net_state_dict[k] = v.cpu()

    shared_dict = mp.Manager().dict()
    shared_dict.update(net_state_dict)

    return shared_dict


def clone_parameters(net: nn.Module) -> List[torch.Tensor]:
    """
    Get the clone of network's parameters. It should be used if your network will change
    and want to get the parameters before it.
    Args:
        net: The network module.

    Returns:
        The cloned parameters.
    """
    cloned_parameters = [p.clone() for p in net.parameters()]
    return cloned_parameters


def check_updated(parameters_before_update: List[torch.Tensor], parameters_after_update: List[torch.Tensor]) -> bool:
    """
    Check whether the network is updated, i.e. parameters changed. Should be used with clone_parameters().

    Examples:
        parameters_before = clone_parameters(net)
        # code to change network
        parameters_after = clone_parameters(net)
        updated = check_updated(parameters_before, parameters_after)

    Args:
        parameters_before_update: Parameters before the update
        parameters_after_update: Parameters after the update

    Returns:
        A boolean value.
    """
    assert len(parameters_before_update) == len(parameters_after_update)
    for p_b, p_a in zip(parameters_before_update, parameters_after_update):
        if not torch.equal(p_b, p_a):
            return True
    return False


def initialize_fc(model: nn.Module):
    """
    Initialize linear with xavier_normal
    Args:
        model: The network module

    Returns:
        No returns.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.01)


def to_device(*args, device="cuda"):
    """
    Send multiple args to device.
    Args:
        *args: The object to sent, could be tensor, nn.Module, etc.
        device: The device to sent to, by default "cuda"

    Returns:
        The objects.
    """
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return tuple(ret)


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl