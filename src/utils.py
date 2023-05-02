#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:utils.py
@time:2023/02/16
"""
import copy
import os
import pickle
import time
from typing import Tuple, Dict, List

import numpy as np
import torch

import rl_cpp
from agent_for_cpp import VecEnvAgent, SingleEnvAgent
from global_vars import DEFAULT_RL_DATASET_DIR, RLDataset
from nets import PolicyNet
import common_utils


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


def sl_net(checkpoint_path: str = r"models/il_net_checkpoint.pth", device: str = "cuda"):
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


def sl_single_env_agent(checkpoint_path: str = r"models/il_net_checkpoint.pth", device: str = "cuda",
                        jit=False):
    """
    Get a supervised learning single env agent.
    Args:
        checkpoint_path: The path to sl net.
        device: The device of the agent.
        jit: Whether convert to script module.

    Returns:

    """
    net = sl_net(checkpoint_path, device)
    agent = SingleEnvAgent(net).to(device)
    if jit:
        agent = torch.jit.script(agent).to(device)
    return agent


def sl_vec_env_agent(checkpoint_path: str = r"models/il_net_checkpoint.pth", device: str = "cuda",
                     jit=False):
    """
    Get a supervised learning single env agent.
    Args:
        checkpoint_path: The path to sl net.
        device: The device of the agent.
        jit: Whether convert to script module.

    Returns:

    """
    net = sl_net(checkpoint_path, device)
    agent = VecEnvAgent(net).to(device)
    if jit:
        agent = torch.jit.script(agent).to(device)
    return agent


def simple_env():
    dataset = load_rl_dataset("valid")
    deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    env = rl_cpp.BridgeBiddingEnv(deal_manager, [1, 1, 1, 1], None, False, True)
    return env


def simple_vec_env(num_envs: int = 10, use_par_score: bool = False, replay_buffer: rl_cpp.ReplayBuffer = None):
    dataset = load_rl_dataset("valid")
    deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    eval_ = True
    if replay_buffer is not None:
        eval_ = False
    vec_env = rl_cpp.BridgeVecEnv()
    for i in range(num_envs):
        env = rl_cpp.BridgeBiddingEnv(deal_manager, [1, 1, 1, 1], replay_buffer, use_par_score, eval_)
        vec_env.push(env)
    return vec_env


def tensor_dict_to_device(tensor_dict: Dict[str, torch.Tensor], device: str):
    ret = {}
    for key, value in tensor_dict.items():
        ret[key] = value.to(device)
    return ret


def analyze(net: PolicyNet, device: str = "cuda"):
    agent = VecEnvAgent(copy.deepcopy(net)).to(device)
    model_locker = rl_cpp.ModelLocker([torch.jit.script(agent).to(device)], device)
    actor = rl_cpp.VecEnvActor(model_locker)
    dataset = load_rl_dataset("valid")
    vec_env = rl_cpp.BridgeVecEnv()
    manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    for i in range(50000):
        env = rl_cpp.BridgeBiddingEnv(manager, [1, 1, 1, 1], None, False, True)
        vec_env.push(env)
    obs = {}
    obs = vec_env.reset(obs)
    while not vec_env.all_terminated():
        reply = actor.act(obs)
        obs, r, t = vec_env.step(reply)
    envs = vec_env.get_envs()
    deal_info = []
    for env in envs:
        state = env.get_state()
        contract = state.get_contract()
        trump = contract.trumps()
        actual_trick, dd_trick = state.get_actual_trick_and_dd_trick()
        deal_info.append((trump, actual_trick, dd_trick))
    with open("analyze.pkl", "wb") as fp:
        pickle.dump(deal_info, fp)
    print("Dump ok.")


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
        _cards, _ddts, _par_scores = dataset["cards"], dataset["ddts"], dataset["par_scores"]
        self.cards = _cards[:num_deals]
        self.ddts = _ddts[:num_deals]
        self.par_scores = _par_scores[:num_deals]
        self.device = eval_device
        self.num_deals = num_deals
        self.num_threads = num_threads
        self.num_deals_per_thread = self.num_deals // self.num_threads
        # when evaluating, all actors choose greedy action
        self.greedy = [1, 1, 1, 1]

        self.vec_env0_list = []
        self.vec_env1_list = []
        for i_t in range(self.num_threads):
            vec_env_0 = rl_cpp.BridgeVecEnv()
            vec_env_1 = rl_cpp.BridgeVecEnv()
            left = i_t * self.num_deals_per_thread
            right = (i_t + 1) * self.num_deals_per_thread
            deal_manager_0 = rl_cpp.BridgeDealManager(
                self.cards[left: right],
                self.ddts[left:right],
                self.par_scores[left:right])
            deal_manager_1 = rl_cpp.BridgeDealManager(
                self.cards[left: right],
                self.ddts[left:right],
                self.par_scores[left:right])
            for i_env in range(self.num_deals_per_thread):
                env_0 = rl_cpp.BridgeBiddingEnv(deal_manager_0, self.greedy,
                                                None, False, True)

                env_1 = rl_cpp.BridgeBiddingEnv(deal_manager_1, self.greedy,
                                                None, False, True)
                vec_env_0.push(env_0)
                vec_env_1.push(env_1)
            self.vec_env0_list.append(vec_env_0)
            self.vec_env1_list.append(vec_env_1)

    def evaluate(self, train_net: PolicyNet, oppo_net: PolicyNet) -> Tuple[float, float, float,
    List[rl_cpp.BridgeVecEnv], List[rl_cpp.BridgeVecEnv]]:
        """
        Evaluate between trained net and opponent net
        Args:
            train_net: The trained policy net
            oppo_net: The opponent's policy net, usually it's a supervised learning net.

        Returns:
            The average imp, the standard error of the mean and elapsed time.
        """
        st = time.perf_counter()
        train_agent = VecEnvAgent(copy.deepcopy(train_net))
        train_agent.to(self.device)
        oppo_agent = VecEnvAgent(copy.deepcopy(oppo_net))
        oppo_agent.to(self.device)
        train_locker = rl_cpp.ModelLocker([torch.jit.script(train_agent)], self.device)
        train_actor = rl_cpp.VecEnvActor(train_locker)
        oppo_locker = rl_cpp.ModelLocker([torch.jit.script(oppo_agent)], self.device)
        oppo_actor = rl_cpp.VecEnvActor(oppo_locker)
        ctx = rl_cpp.Context()

        for i_t in range(self.num_threads):
            eval_thread = rl_cpp.VecEnvEvalThreadLoop(train_actor,
                                                      oppo_actor,
                                                      self.vec_env0_list[i_t],
                                                      self.vec_env1_list[i_t])
            ctx.push_thread_loop(eval_thread)
        ctx.start()
        while not ctx.terminated():
            time.sleep(0.5)
        scores_ns = np.concatenate([vec_env_ns.get_returns(0) for vec_env_ns in self.vec_env0_list])
        scores_ew = np.concatenate([vec_env_ew.get_returns(0) for vec_env_ew in self.vec_env1_list])
        imps = [rl_cpp.get_imp(score_ns, score_ew) for score_ns, score_ew in zip(scores_ns, scores_ew)]
        ed = time.perf_counter()
        avg, sem = common_utils.get_avg_and_sem(imps)
        return avg, sem, ed - st, self.vec_env0_list, self.vec_env1_list
