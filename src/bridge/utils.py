#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:utils.py
@time:2023/02/16
"""
import os
import time
from typing import Tuple

import numpy as np
import torch
import rl_cpp
from src.bridge.agent_for_cpp import VecEnvAgent, SingleEnvAgent
from src.bridge.nets import PolicyNet
from src.common_utils.array_utils import multiple_shuffle, get_avg_and_sem
from src.bridge.global_vars import DEFAULT_RL_DATASET_DIR


def load_rl_dataset(usage: str, flatten: bool = True, shuffle: bool = False, dataset_dir: str = DEFAULT_RL_DATASET_DIR) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset
    Args:
        flatten (bool): whether the ddt should be 2 dim
        usage (str): should be one of "train", "valid", "test" or "vs_wb5"
        shuffle (bool): whether to shuffle, False default
        dataset_dir (str): the dir to dataset, the file names should be usage + _trajectories.npy / _ddts.npy

    Returns:
        Tuple[np.ndarray, np.ndarray]: The trajectories and ddts

    """
    assert usage in ["train", "valid", "test", "vs_wb5"]
    trajectories = np.load(os.path.join(dataset_dir, f"{usage}_trajectories.npy"))
    ddts = np.load(os.path.join(dataset_dir, f"{usage}_ddts.npy"))
    if shuffle:
        trajectories, ddts = multiple_shuffle(trajectories, ddts)
    if flatten:
        ddts = ddts.reshape(-1, 20)
    return trajectories, ddts


def sl_net(checkpoint_path: str = r"D:\RL\bridge_research\src\models\il_net_checkpoint.pth", device: str = "cuda"):
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
        _cards, _ddts = load_rl_dataset("valid", flatten=True)
        self.cards = _cards[:num_deals]
        self.ddts = _ddts[:num_deals]
        self.device = eval_device
        self.num_deals = num_deals
        self.num_threads = num_threads
        self.num_deals_per_threads = self.num_deals // self.num_threads
        # when evaluating, all actors choose greedy action
        self.greedy = [1, 1, 1, 1]

    def evaluate(self, _train_net: PolicyNet, _oppo_net: PolicyNet) -> Tuple[float, float]:
        """
        Evaluate between a trained policy net and opponent policy net.
        Args:
            _train_net: The trained policy net.
            _oppo_net: The opponent's policy net.

        Returns:
            The average and standard error of mean of imps.
        """
        _train_agent = VecEnvAgent(_train_net)
        _train_agent.to(self.device)
        _oppo_agent = VecEnvAgent(_oppo_net)
        _oppo_agent.to(self.device)
        _train_locker = rl_cpp.ModelLocker([torch.jit.script(_train_agent)], self.device)
        train_actor = rl_cpp.VecEnvActor(_train_locker)
        _oppo_locker = rl_cpp.ModelLocker([torch.jit.script(_oppo_agent)], self.device)
        oppo_actor = rl_cpp.VecEnvActor(_oppo_locker)
        imp_vec = rl_cpp.IntConVec()
        ctx = rl_cpp.Context()
        for i_t in range(self.num_threads):
            vec_env_0 = rl_cpp.BridgeVecEnv()
            vec_env_1 = rl_cpp.BridgeVecEnv()
            for i_env in range(self.num_deals_per_threads):
                _left = i_env + i_t * self.num_deals_per_threads
                # print(_left)
                _right = _left + 1
                _env_0 = rl_cpp.BridgeBiddingEnv(self.cards[_left:_right],
                                                 self.ddts[_left:_right], self.greedy)
                # envs_0.append(_env_0)

                _env_1 = rl_cpp.BridgeBiddingEnv(self.cards[_left:_right],
                                                 self.ddts[_left:_right], self.greedy)
                # envs_1.append(_env_1)
                vec_env_0.append(_env_0)
                vec_env_1.append(_env_1)
            eval_thread = rl_cpp.VecEvalThreadLoop(vec_env_0,
                                                   vec_env_1,
                                                   train_actor,
                                                   oppo_actor,
                                                   imp_vec,
                                                   False, 1)
            ctx.push_thread_loop(eval_thread)
        ctx.start()
        while not ctx.terminated():
            time.sleep(0.5)
        imps = imp_vec.get_vector()
        return get_avg_and_sem(imps)


class SingleProcessEvaluator:
    def __init__(self, num_deals: int, eval_device: str):
        _cards, _ddts = load_rl_dataset("valid", flatten=True)
        self.cards = _cards[:num_deals]
        self.ddts = _ddts[:num_deals]
        self.device = eval_device
        self.num_deals = num_deals
        # when evaluating, all actors choose greedy action
        self.greedy = [1, 1, 1, 1]

    def evaluate(self, _train_net: PolicyNet, _oppo_net: PolicyNet):
        _train_agent = VecEnvAgent(_train_net)
        _train_agent.to(self.device)
        _oppo_agent = VecEnvAgent(_oppo_net)
        _oppo_agent.to(self.device)
        _train_locker = rl_cpp.ModelLocker([torch.jit.script(_train_agent)], self.device)
        train_actor = rl_cpp.VecEnvActor(_train_locker)
        _oppo_locker = rl_cpp.ModelLocker([torch.jit.script(_oppo_agent)], self.device)
        oppo_actor = rl_cpp.VecEnvActor(_oppo_locker)
        imp_vec = rl_cpp.IntConVec()
        vec_env_0 = rl_cpp.BridgeVecEnv()
        vec_env_1 = rl_cpp.BridgeVecEnv()
        envs_0 = []
        envs_1 = []
        for i_env in range(self.num_deals):
            _left = i_env

            _right = _left + 1
            _env_0 = rl_cpp.BridgeBiddingEnv(self.cards[_left:_right],
                                             self.ddts[_left:_right], self.greedy)
            _env_1 = rl_cpp.BridgeBiddingEnv(self.cards[_left:_right],
                                             self.ddts[_left:_right], self.greedy)
            envs_0.append(_env_0)
            envs_1.append(_env_1)
            vec_env_0.append(_env_0)
            vec_env_1.append(_env_1)
        eval_thread = rl_cpp.VecEvalThreadLoop(vec_env_0,
                                               vec_env_1,
                                               train_actor,
                                               oppo_actor,
                                               imp_vec,
                                               False, 1)
        eval_thread.main_loop()
        imps = imp_vec.get_vector()
        scores_0 = [env.returns()[0] for env in envs_0]
        scores_1 = [env.returns()[0] for env in envs_1]
        imps_ = [rl_cpp.get_imp(int(score_0), int(score_1)) for score_0, score_1 in zip(scores_0, scores_1)]
        print(get_avg_and_sem(imps_))
        return get_avg_and_sem(imps)


class ImpEvaluator:
    """A evaluator using imp env"""

    def __init__(self, num_deals: int, eval_device: str):
        self.num_deals = num_deals
        self.device = eval_device
        _cards, _ddts = load_rl_dataset("valid", flatten=True)
        self.cards = _cards[:num_deals]
        self.ddts = _ddts[:num_deals]

    def evaluate(self, train_net: PolicyNet, oppo_net: PolicyNet):
        train_agent = SingleEnvAgent(train_net)
        train_agent.to(self.device)
        oppo_agent = SingleEnvAgent(oppo_net)
        oppo_agent.to(self.device)
        train_locker = rl_cpp.ModelLocker([torch.jit.script(train_agent)], self.device)
        oppo_locker = rl_cpp.ModelLocker([torch.jit.script(oppo_agent)], self.device)
        train_actor = rl_cpp.SingleEnvActor(train_locker, 0, 1.0, True)
        oppo_actor = rl_cpp.SingleEnvActor(oppo_locker, 0, 1.0, True)
        actors = [train_actor, oppo_actor, train_actor, oppo_actor]
        env = rl_cpp.ImpEnv(self.cards, self.ddts, [1, 1, 1, 1], True)
        eval_thread = rl_cpp.EvalImpThreadLoop(actors, env, self.num_deals)
        eval_thread.main_loop()
        imps = env.history_imps()
        return get_avg_and_sem(imps)
