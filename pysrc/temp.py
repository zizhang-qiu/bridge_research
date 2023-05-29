#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:temp.py
@time:2023/02/15
"""
import argparse
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
import dds
import rl_cpp
import numpy as np
import torch.nn.functional as F
import torchviz
from torchviz import make_dot
import torchview
from agent_for_cpp import SingleEnvAgent, VecEnvAgent
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
    # imps = np.load("vs_wbridge5/folder_28/imps.npy")
    # c = Counter(imps)
    # print(c)
    # print(sorted(c.items()))
    # torch.set_printoptions(threshold=1000000)
    # common_utils.set_random_seeds(1)
    # dataset = load_rl_dataset("train")
    # deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    # deal = deal_manager.next()
    # state = rl_cpp.BridgeBiddingState(deal)
    # obs = rl_cpp.make_obs_tensor_dict(state, 1)
    # for k, v in obs.items():
    #     print(k, v.shape)
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
    # rl_cpp.accessor_test()
    # device = "cuda"
    # net = sl_net()
    #
    # s_agent = SingleEnvAgent(net).to(device)
    # v_agent = VecEnvAgent(net).to(device)
    # s_actor = rl_cpp.SingleEnvActor(rl_cpp.ModelLocker([torch.jit.script(s_agent).to(device)], device))
    # v_actor = rl_cpp.VecEnvActor(rl_cpp.ModelLocker([torch.jit.script(v_agent).to(device)], device))
    #
    # deal = deal_manager.next()
    # state = rl_cpp.BridgeBiddingState(deal)
    # params = rl_cpp.SearchParams()
    # params.verbose_level = 1
    # params.max_rollouts = 1000
    # params.min_rollouts = 100
    # params.max_particles = 100000
    # state.apply_action(6)
    # state.apply_action(13)
    # state.apply_action(0)
    # state.apply_action(14)
    #

    # searcher = rl_cpp.Searcher(params, [v_actor for _ in range(NUM_PLAYERS)], 100)
    # while not state.terminated():
    #
    #     obs = rl_cpp.make_obs_tensor_dict(state, 1)
    # # print(obs)
    #     reply = s_actor.act(obs)
    #     # print(reply)
    #     probs = torch.exp(reply["log_probs"])
    #     st = time.perf_counter()
    #     action = searcher.search(state, probs)
    #     ed = time.perf_counter()
    #     print(action, f"Elapsed time: {ed - st}", sep="\t")
    #     state.apply_action(action)
    #     input("press random key.")
    # print(state)
    # cards = dataset["cards"]
    # ddts = dataset["ddts"]
    # par_scores = dataset["par_scores"]
    # with open("temp.txt", "w") as f:
    #     cards_str = np.array2string(par_scores[:100], separator=", ", threshold=sys.maxsize, max_line_width=sys.maxsize)
    #     cards_str = cards_str.replace("[", "{").replace("]", "}")
    #     f.write(cards_str)
    # a = torch.rand(10)
    # a = F.log_softmax(a, -1)
    # print(a)
    # a = F.log_softmax(a, -1)
    # print(a)
    # print(torch.exp(a))
    # probs = torch.tensor([5.0048e-01, 0.0000e+00, 0.0000e+00, 1.5615e-03, 8.6133e-04, 1.9591e-04,
    #                       1.5676e-06, 4.2391e-06, 2.1365e-07, 9.7709e-07, 3.2186e-07, 5.4442e-11,
    #                       4.4298e-09, 9.7094e-12, 6.0384e-14, 6.9944e-15, 1.9726e-15, 2.9187e-14,
    #                       1.8934e-18, 1.4101e-23, 1.7813e-24, 9.5524e-29, 4.3005e-34, 6.8552e-31,
    #                       2.6937e-32, 3.2111e-33, 2.1944e-35, 2.2660e-15, 8.7025e-36, 1.2179e-34,
    #                       5.9316e-34, 5.1651e-35, 4.2461e-31, 1.3505e-23, 2.7572e-13, 1.4730e-10,
    #                       1.4084e-31, 2.5270e-12])
    # values = torch.zeros(38, dtype=torch.float)
    # values[0] = -112280
    # values[3] = -156770
    # values[4] = -142790
    # values[5] = -161930
    # probs_posterior = probs * torch.exp(values / (100 * math.sqrt(1000)))
    # print(probs_posterior)
    # print(probs_posterior / torch.sum(probs_posterior))
    # imps = np.load("vs_wbridge5/folder_47/imps_0.npy")
    # print(imps)
    # print(np.load("vs_wbridge5/folder_44/imps_0.npy")[:imps.size])
    net = sl_net(device="cpu")
    torch.save(net.state_dict(), "net.pth")
