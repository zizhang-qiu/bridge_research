#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:temp.py
@time:2023/02/15
"""
import argparse
import copy
import json
import math
import os
import pickle
import pprint
import re
import sys
import time
from collections import Counter
from typing import Dict, Protocol, List
import torch
import yaml
from torch import nn
from torch.distributions import Categorical

import dds
import rl_cpp
import numpy as np
import torch.nn.functional as F
import torchviz
from torchviz import make_dot
import torchview
from agent_for_cpp import SingleEnvAgent, VecEnvAgent, EnsembleAgent
from bridge_consts import NUM_SUITS, PLUS_MINUS_SYMBOL, NUM_CARDS, NUM_PLAYERS
from global_vars import RLDataset
from nets import PolicyNet, PolicyNet2, PolicyNetRelu
import common_utils
from utils import simple_env, sl_net, sl_single_env_agent, sl_vec_env_agent, simple_vec_env, Evaluator, load_rl_dataset, \
    tensor_dict_to_device, analyze, sl_net2


def json_2_np(path):
    with open(path) as fp:
        obj = json.load(fp)
    return np.array(obj)


def merge_array_dict(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]):
    assert d1.keys() == d2.keys()
    ret = {}
    for key in d1.keys():
        if d1[key].ndim == 2:
            ret[key] = np.vstack([d1[key], d2[key]])
        else:
            ret[key] = np.hstack([d1[key], d2[key]])
    return ret


def convert(usage, save_path="dataset/rl_data"):
    trajectory_path = rf"D:\RL\rlul\pyrlul\bridge\dataset\rl_data\{usage}_trajectories"
    ddt_path = rf"D:\RL\rlul\pyrlul\bridge\dataset\rl_data\{usage}_ddts"

    traj_np = json_2_np(trajectory_path)
    ddt_np = json_2_np(ddt_path).reshape([-1, 5, 4])
    np.save(os.path.join(save_path, f"{usage}_trajectories.npy"), traj_np)
    np.save(os.path.join(save_path, f"{usage}_ddts.npy"), ddt_np)


def _deal_trajectory(line: str) -> List[int]:
    actions = [int(action) for action in line.split(' ')]
    return actions[:NUM_CARDS]


def find_par_zero_deal():
    while True:
        flag = False
        cards = np.zeros([32, NUM_CARDS], dtype=int)
        for i in range(32):
            cards[i] = np.random.permutation(NUM_CARDS)
        ddt, pres = dds.calc_all_tables(cards, False)
        for i, par in enumerate(pres):
            par_score_str = par.parScore[1].value.decode("utf-8")
            par_score = int(re.search(r"[-]?\d+", par_score_str).group())
            # print(par_score)
            if par_score == 0:
                print(cards[i])
                print(par.parContractsString[0].value.decode("utf-8"))
                flag = True
                break
        if flag:
            break


def load_models_from_directory(directory: str) -> List[nn.Module]:
    model_paths = common_utils.find_files_in_dir(directory, "model", 0)
    models = []
    for path in model_paths:
        net = PolicyNet2()
        net.load_state_dict(torch.load(path)["model_state_dict"]["policy"])
        models.append(net)
    return models


if __name__ == '__main__':
    # torch.set_printoptions(threshold=1000000)
    # common_utils.set_random_seeds(1)
    # dataset = load_rl_dataset("valid")

    search_imps = np.load("vs_wbridge5/folder_60/imps_0.npy")
    original_imps = np.load("vs_wbridge5/folder_58/imps.npy")[:search_imps.size]
    print(search_imps)
    print(original_imps)
    print(common_utils.get_avg_and_sem(original_imps))
    print(common_utils.get_avg_and_sem(search_imps))

