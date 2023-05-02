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
import time
from typing import Dict, Protocol, List
import torch
import yaml

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


if __name__ == '__main__':
    torch.set_printoptions(threshold=100000)

    # net = sl_net(device="cuda")
    # agent = SingleEnvAgent(net)
    # dataset = load_rl_dataset("train")
    # manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    # deal = manager.next()
    # state = rl_cpp.BridgeBiddingState(deal)
    # for action in [0, 3, 0, 0, 0]:
    #     state.apply_action(action)
    # print(state)
    # print(state.get_actual_trick_and_dd_trick())
    # net = PolicyNet()
    # net.load_state_dict(torch.load("a2c/folder_5/checkpoint_1.pth")["model_state_dict"]["policy"])
    # analyze(net, "cuda")
    # with open("analyze.pkl", "rb") as fp:
    #     deal_info = pickle.load(fp)
    # print(deal_info)
    rl_cpp.generate_deals(1000, 42)
