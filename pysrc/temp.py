#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:temp.py
@time:2023/02/15
"""
import argparse
import json
import os
import pickle
import pprint
import re
import time
from typing import Dict, Protocol, List
import torch
import yaml
import dds
import rl_cpp
import numpy as np

import torchviz
from torchviz import make_dot
import torchview
from agent_for_cpp import SingleEnvAgent
from bridge_vars import NUM_SUITS, PLUS_MINUS_SYMBOL, NUM_CARDS, NUM_PLAYERS
from global_vars import RLDataset
from nets import PolicyNet
import common_utils
from utils import simple_env, sl_net, sl_single_env_agent, sl_vec_env_agent, simple_vec_env, Evaluator, load_rl_dataset, \
    tensor_dict_to_device, analyze


def json_2_np(path):
    with open(path) as fp:
        obj = json.load(fp)
    return np.array(obj)


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


def make_open_spiel_test_data():
    data_path = r"D:\Projects\bridge_research\dataset\expert\test.txt"
    with open(data_path, "r") as f:
        deals = f.readlines()
    ret_deals = [_deal_trajectory(deal) for deal in deals]
    deals_np = np.array(ret_deals)
    assert np.array_equal(deals_np.shape, [10000, NUM_CARDS])
    np.save(r"D:\Projects\bridge_research\dataset\rl_data\vs_wb5_open_spiel_trajectories.npy", deals_np)


def parse_args():
    """
    Parse arguments using Argument parser
    Returns:
        The args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--policy_lr", type=float, default=1e-6)
    parser.add_argument("--value_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--buffer_capacity", type=int, default=800000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--sample_batch_size", type=int, default=2048)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--entropy_ratio", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--actor_update_freq", type=int, default=100)
    parser.add_argument("--opponent_update_freq", type=int, default=400)
    parser.add_argument("--save_dir", type=str, default="a2c")

    return parser.parse_args()


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


if __name__ == '__main__':
    torch.set_printoptions(threshold=100000)
    np.set_printoptions(threshold=100000)
    common_utils.set_random_seeds(1)
    cards = np.zeros([101, NUM_CARDS], dtype=int)
    for i in range(101):
        cards[i] = np.random.permutation(NUM_CARDS)
    dd_table_res_list, pres_list = dds.calc_all_tables(cards)
    ddts = dds.get_ddts_from_dd_table_res_list(dd_table_res_list)
    print(ddts)
    print(ddts.shape)
    dds.get_par_scores_and_contracts_from_pres_list(pres_list)
