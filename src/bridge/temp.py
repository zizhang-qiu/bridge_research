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
from typing import Dict

import numpy as np

from src.bridge.bridge_vars import NUM_SUITS
from src.bridge.pbn import get_trajectories_and_ddts_from_pbn_file


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
    # net = PolicyNet()
    # supervised_net = sl_net()
    # evaluator = Evaluator(10000, 8, "cuda")
    # for i in range(10):
    #     checkpoint_path = os.path.join(r"D:\RL\bridge_research\src\policy_gradient\20230321190524", f"checkpoint_{i}.pth")
    #     net.load_state_dict(torch.load(checkpoint_path)["model_state_dict"]["policy"])
    #     avg, sem = evaluator.evaluate(net, supervised_net)
    #     print(i, avg, sem)
    # opt = Adan(net.parameters(), lr=1e-3)
    # print(opt.state_dict())
    # print(net.state_dict())
    # cards, ddts = get_trajectories_and_ddts_from_pbn_file("../dataset/pbn/example.pbn")
    # print(cards, ddts)
    with open(r"D:\RL\bridge_research\src\dataset\rl_data\valid.pkl", "rb") as f:
        dataset:Dict = pickle.load(f)

    print(dataset)
    for key, value in dataset.items():
        print(value.shape)