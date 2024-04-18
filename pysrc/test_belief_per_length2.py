"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: test_belief_per_length2.py.py
@time: 2024/2/29 20:41
"""
import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import rl_cpp
from belief_model import BeliefModel
from agent_for_cpp import VecEnvAgent
from nets import PolicyNet2
from utils import tensor_dict_to_device
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Test Belief Per-Length")
    parser.add_argument("--num_belief_per_length", type=int, default=1000)
    parser.add_argument("--num_total_nodes", type=int, default=100000)
    parser.add_argument("--belief_path", type=str, default="belief/folder_4/latest.pth")
    parser.add_argument(
        "--policy_path", type=str, default="a2c_fetch/4/folder_10/model2.pth"
    )
    parser.add_argument("--num_random_sample", type=int, default=1000)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="belief_quality")


def main():
    args = parse_args()
    print(vars(args))
    belief_net = BeliefModel()
    belief_net.load_state_dict(torch.load(args.belief_path))
    belief_net.to("cuda")

    policy_net = PolicyNet2()
    # print(torch.load(args.policy_path))
    policy_net.load_state_dict(
        torch.load(args.policy_path)["model_state_dict"]["policy"]
    )
    policy_net.to("cuda")

    agent = VecEnvAgent(policy_net).to("cuda")
    deal_manager = rl_cpp.RandomDealManager(1)

    env = rl_cpp.BridgeBiddingEnv(deal_manager, [0, 0, 0, 0])
