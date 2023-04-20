#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:global_vars.py
@time:2023/02/16
"""
from collections import namedtuple
from enum import IntEnum
from typing import List, Union, Tuple, Dict

import numpy as np

Vector = Union[np.ndarray, List, Tuple]
Action = int
RLDataset = Dict[str, np.ndarray]


class PlayerId(IntEnum):
    CHANCE_PLAYER = -1
    TERMINAL_PLAYER = -2
    INVALID_PLAYER = -3


PlayerAction = namedtuple("PlayerAction", ["player", "action"])

DEFAULT_RL_DATASET_DIR = r"../dataset/rl_data"
