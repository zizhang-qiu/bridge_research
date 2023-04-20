#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:torch_utils.py
@time:2023/02/17
"""
from typing import List

import torch
import torch.multiprocessing as mp
from torch import nn

from src.common_utils.assert_utils import assert_eq


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
    assert_eq(len(parameters_before_update), len(parameters_after_update))
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
