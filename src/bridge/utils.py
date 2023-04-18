#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:utils.py
@time:2023/02/16
"""
import os
import pickle
import time
from typing import Tuple

import torch

import rl_cpp
from src.bridge.agent_for_cpp import VecEnvAgent, SingleEnvAgent
from src.bridge.global_vars import DEFAULT_RL_DATASET_DIR, RLDataset
from src.bridge.nets import PolicyNet
from src.common_utils.array_utils import get_avg_and_sem


def load_rl_dataset(usage: str, dataset_dir: str = DEFAULT_RL_DATASET_DIR) \
        -> RLDataset:
    """
    Load dataset.
    Args:
        usage (str): should be one of "train", "valid", "vs_wb5_fb" or "vs_wb5_open_spiel"
        dataset_dir (str): the dir to dataset, the file names should be usage + _trajectories.npy / _ddts.npy

    Returns:
        RLDataset: The cards, ddts and par scores, combined as a dict
    """
    dataset_name = usage + ".pkl"
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"No such path: {dataset_path}, please check path or name.")

    with open(dataset_path, "rb") as fp:
        dataset: RLDataset = pickle.load(fp)

    return dataset


def sl_net(checkpoint_path: str = r"..\..\models\il_net_checkpoint.pth", device: str = "cuda"):
    """
    Get a supervised learning policy net.
    Args:
        checkpoint_path: The path to the checkpoint
        device: The device

    Returns:
        The net.
    """
    net = PolicyNet()
    net.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    net.to(device)
    return net


class Evaluator:
    def __init__(self, num_deals: int, num_threads: int, eval_device: str):
        """
        An evaluator for evaluating agents.
        Args:
            num_deals: The number of deals.
            num_threads: The number of threads.
            eval_device: The device for the actors.
        """
        assert num_deals % num_threads == 0
        dataset = load_rl_dataset("valid")
        _cards, _ddts = dataset["cards"], dataset["ddts"]
        self.cards = _cards[:num_deals]
        self.ddts = _ddts[:num_deals]
        self.device = eval_device
        self.num_deals = num_deals
        self.num_threads = num_threads
        self.num_deals_per_threads = self.num_deals // self.num_threads
        # when evaluating, all actors choose greedy action
        self.greedy = [1, 1, 1, 1]

        self.imp_vec = rl_cpp.IntConVec()

        self.vec_env0_list = []
        self.vec_env1_list = []
        for i_t in range(self.num_threads):
            vec_env_0 = rl_cpp.BridgeVecEnv()
            vec_env_1 = rl_cpp.BridgeVecEnv()
            for i_env in range(self.num_deals_per_threads):
                left = i_env + i_t * self.num_deals_per_threads
                right = left + 1
                env_0 = rl_cpp.BridgeBiddingEnv(self.cards[left:right],
                                                self.ddts[left:right], self.greedy)

                env_1 = rl_cpp.BridgeBiddingEnv(self.cards[left:right],
                                                self.ddts[left:right], self.greedy)
                vec_env_0.append(env_0)
                vec_env_1.append(env_1)
            self.vec_env0_list.append(vec_env_0)
            self.vec_env1_list.append(vec_env_1)

    def evaluate(self, train_net: PolicyNet, oppo_net: PolicyNet) -> Tuple[float, float, float]:
        """
        Evaluate between trained net and opponent net
        Args:
            train_net: The trained policy net
            oppo_net: The opponent's policy net, usually it's a supervised learning net.

        Returns:
            The average imp, the standard error of the mean and elapsed time.
        """
        st = time.perf_counter()
        train_agent = VecEnvAgent(train_net)
        train_agent.to(self.device)
        oppo_agent = VecEnvAgent(oppo_net)
        oppo_agent.to(self.device)
        train_locker = rl_cpp.ModelLocker([torch.jit.script(train_agent)], self.device)
        train_actor = rl_cpp.VecEnvActor(train_locker)
        oppo_locker = rl_cpp.ModelLocker([torch.jit.script(oppo_agent)], self.device)
        oppo_actor = rl_cpp.VecEnvActor(oppo_locker)
        ctx = rl_cpp.Context()

        for i_t in range(self.num_threads):
            eval_thread = rl_cpp.VecEvalThreadLoop(self.vec_env0_list[i_t],
                                                   self.vec_env1_list[i_t],
                                                   train_actor,
                                                   oppo_actor,
                                                   self.imp_vec,
                                                   False, 1)
            ctx.push_thread_loop(eval_thread)
        ctx.start()
        while not ctx.terminated():
            time.sleep(0.5)
        imps = self.imp_vec.get_vector()
        assert len(imps) == self.num_deals
        self.imp_vec.clear()
        ed = time.perf_counter()
        avg, sem = get_avg_and_sem(imps)
        return avg, sem, ed - st

