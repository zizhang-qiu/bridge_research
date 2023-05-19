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
    # torch.set_printoptions(threshold=1000000)
    # common_utils.set_random_seeds(1)
    # dataset = load_rl_dataset("train")
    # deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    # agent = sl_single_env_agent()
    # agent = sl_vec_env_agent()
    # env = rl_cpp.ImpEnv(deal_manager, [0, 0, 0, 0])
    # obs = env.reset()
    # transition_buffers = [rl_cpp.BridgeTransitionBuffer() for _ in range(4)]
    # while not env.terminated():
    #     player = env.get_acting_player()
    #     reply = agent.act(tensor_dict_to_device(obs, "cuda"))
    #     print(reply)
    #     transition_buffers[player].push_obs_and_reply(obs, reply)
    #     obs, r, t = env.step(reply)
    #
    # reward = env.returns()
    # for i, transition_buffer in enumerate(transition_buffers):
    #     transitions, weight = transition_buffer.pop_transitions(reward[i])
    #     for transition in transitions:
    #         print(transition.to_dict())
    # evaluator = Evaluator(100, 1, "cuda")
    # net1 = sl_net()
    # net2 = sl_net()
    # avg, sem, t, env0, env1 = evaluator.evaluate(net1, net2)
    # print(avg, sem, t)
    # env_ns = env0[0].get_envs()
    # env_ew = env1[0].get_envs()
    # for i in range(100):
    #     print(env_ns[i], env_ew[i])
    # replay_buffer = rl_cpp.Replay(800000, 42, 0.6, 0.4, 0)
    # imp_vec_env = rl_cpp.ImpVecEnv()
    # model_locker = rl_cpp.ModelLocker([torch.jit.script(agent)], "cuda")
    # actor = rl_cpp.VecEnvActor(model_locker)
    # for i in range(10):
    #     imp_env = rl_cpp.ImpEnvWrapper(deal_manager, [0, 0, 0, 0], replay_buffer)
    #     imp_vec_env.push(imp_env)
    #
    # imp_vec_env.reset()
    # print(imp_vec_env.get_feature())
    # while not imp_vec_env.all_terminated():
    #     obs = imp_vec_env.get_feature()
    #     print(obs)
    #     reply = actor.act(obs)
    #     imp_vec_env.step(reply)
    #
    # t = rl_cpp.ImpThreadLoop(imp_vec_env, actor)
    # context = rl_cpp.Context()
    # context.push_thread_loop(t)
    # context.start()
    # while replay_buffer.size() < 10000:
    #     print(replay_buffer.size())
    #     time.sleep(1)
    #
    # transition, weight = replay_buffer.sample(10, "cuda")
    # print(transition.to_dict()["s"])
    # print(transition.reply["values"])
    # print(transition.reward)
    # print(weight)
    # priority = agent.compute_priority(transition)
    # print(priority)

    # imp_vec_env.reset()
    # obs = imp_vec_env.get_feature()
    # print(obs["perfect_s"].shape)
    # while not imp_vec_env.all_terminated():
    #     reply = agent.act(tensor_dict_to_device(obs, "cuda"))
    #     print(reply["a"])
    #     imp_vec_env.step(reply)
    #     obs = imp_vec_env.get_feature()
    # print(replay_buffer.num_add())
    # print(replay_buffer.size())
    # transition, weight = replay_buffer.sample(10, "cuda")
    # print(transition.to_dict())
    # print(weight)
    # priority = agent.compute_priority(transition)
    # print(priority)
    # save_dir = "vs_wbridge5/folder_14"
    # logs = []
    # for i in range(8):
    #     with open(os.path.join(save_dir, f"log_{i}.txt"), "r") as f:
    #         log = f.read()
    #     logs.append(log)
    # with open(os.path.join(save_dir, "log.txt"), "a") as f:
    #     f.write("\n\n".join(logs))
    # save_dir = "vs_wbridge5/folder_16"
    # imps_list = [np.load(os.path.join(save_dir, f"imps_{i}.npy")) for i in range(8)]
    # imps = np.concatenate(imps_list)
    #
    # final_imps = np.concatenate([imps, np.load("vs_wbridge5/folder_19/imps.npy")])
    # print(final_imps.shape)
    # avg, sem = common_utils.get_avg_and_sem(final_imps)
    # print(avg, sem)
    # deal = deal_manager.next()
    # state = rl_cpp.BridgeBiddingState(deal)
    # print(state)
    # evaluations = state.get_hand_evaluation()
    # for evaluation in evaluations:
    #     print(evaluation)
    #     print(evaluation.length_per_suit)
    # imps = []
    # for i in range(20, 25):
    #     imp = np.load(os.path.join(f"vs_wbridge5/folder_{i}", "imps.npy"))
    #     imps.append(imp)
    # final_imps = np.concatenate(imps)
    # print(final_imps.shape)
    # print(common_utils.get_avg_and_sem(final_imps))
    # stats = torch.load("imitation_learning/metrics/stats.pth")
    # for k, v in stats.items():
    #     print(k, v)
    pass
