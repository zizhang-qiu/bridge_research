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
from utils import simple_env, sl_net, sl_single_env_agent, sl_vec_env_agent, simple_vec_env, Evaluator, load_rl_dataset


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
    # imps_dir = "../vs_wbridge5/20230329101525"
    # imps_list = [np.load(os.path.join(imps_dir, f"imps_{i}.npy")) for i in range(8)]
    # for imps_np in imps_list:
    #     print(imps_np.shape)
    # imps = np.concatenate(imps_list)
    # np.set_printoptions(threshold=100000)
    # print(imps)
    # print(imps.shape)
    # print(get_avg_and_sem(imps))
    # print(np.sum(imps))
    # checkpoint = torch.load(r"D:\Projects\bridge_research\src\policy_gradient\20230327101038\checkpoint_0.pth")
    # pprint.pprint(checkpoint["optimizer_state_dict"]["policy"])
    # trained_net = PolicyNet()
    # evaluator = Evaluator(50000, 8, "cuda")
    # for i in range(10):
    #     trained_net.load_state_dict(torch.load(f"../../policy_gradient/20230327101038/checkpoint_{i}.pth")
    #                                 ["model_state_dict"]["policy"])
    #     supervised_net = sl_net()
    #     avg, sem = evaluator.evaluate(trained_net, supervised_net)
    #     print(f"checkpoint {i}, result is {avg}{PLUS_MINUS_SYMBOL}{sem}")
    # cards, ddts = load_rl_dataset("train", flatten=True)
    # calc_ddts, calc_par_scores = rl_cpp.calc_all_tables(cards[:10000])
    # print(np.array_equal(calc_ddts, ddts[:10000]))
    # avg_imps = np.load(r"D:\Projects\bridge_research\policy_gradient\20230322193357\avg_imp.npy")
    # sem = np.load(r"D:\Projects\bridge_research\policy_gradient\20230322193357\sem_imp.npy")
    # print(avg_imps[660:700], sem[660:700])
    # cards = np.load(r"D:\Projects\bridge_research\dataset\rl_data\vs_wb5_open_spiel_trajectories.npy")
    # # print(cards[:10])
    # ddts, par_scores = dds.calc_all_tables(cards)
    # pars = np.zeros(10000, dtype=int)
    # for i, par in enumerate(par_scores):
    #     par_score = get_par_score_from_par_results(par, view=0)
    #     pars[i] = par_score
    # dataset = {
    #     "cards": cards,
    #     "ddts": ddts,
    #     "par_scores": pars
    # }
    #
    # with open(os.path.join(r"D:\Projects\bridge_research\dataset\rl_data", f"vs_wb5_open_spiel.pkl"), "wb") as f:
    #     pickle.dump(dataset, f)

    # with open("dataset/rl_data/train.pkl", "rb") as fp:
    #     dataset: RLDataset = pickle.load(fp)

    # print(dataset["par_scores"][:100])
    # env = rl_cpp.BridgeBiddingEnv2(dataset["cards"], dataset["ddts"], dataset["par_scores"], [1, 1, 1, 1])
    # net = sl_net(device="cuda")
    # agent = SingleEnvAgent(net)
    # obs = env.reset()
    # while not env.terminated():
    #     obs = obs.to("cuda")
    #     action, _, _ = agent.act(obs)
    #     print(action)
    #     obs, r, t = env.step(action)
    # print(env)
    # print(dataset["par_scores"][0])
    # print(env.returns())
    # deal = rl_cpp.BridgeDeal()
    # deal.cards = dataset["cards"][0]
    # # deal.ddt = dataset["ddts"][0]
    # state = rl_cpp.BridgeBiddingState(deal)
    # print(state)
    # for a in [3, 0, 0, 0]:
    #     state.apply_action(a)
    # print(state)
    # print(torch.cuda.device_count())
    # log_probs = torch.log(torch.tensor([0.0, 0.3, 0.1, 0.6]))
    # print(log_probs)
    # legal_actions = torch.ones(38)
    # print(torch.multinomial(legal_actions, 1))
    # print(torch.multinomial(legal_actions, 1).squeeze())
    # print(rl_cpp.check_prob_not_zero(torch.tensor([1]), log_probs))
    # jit_agent = sl_single_env_agent(device="cuda", jit=True)
    # model_locker = rl_cpp.ModelLocker([jit_agent], "cuda")
    # actor = rl_cpp.SingleEnvActor(model_locker)
    # env = simple_env()
    # obs = env.reset()
    # print(obs)
    # reply = actor.act(obs)
    # print(reply)
    common_utils.set_random_seeds(1)
    st = time.perf_counter()
    jit_agent = sl_vec_env_agent(jit=True)
    model_locker = rl_cpp.ModelLocker([jit_agent], "cuda")
    vec_actor = rl_cpp.VecEnvActor(model_locker)
    # vec_env_ns = simple_vec_env(50000)
    # vec_env_ew = simple_vec_env(50000)
    # t = rl_cpp.VecEnvEvalThreadLoop(vec_actor, vec_actor, vec_env_ns, vec_env_ew)
    # t.main_loop()
    # scores_ns = vec_env_ns.get_returns(0)
    # scores_ew = vec_env_ew.get_returns(0)
    # # print(scores_ns, scores_ew)
    # imps = [rl_cpp.get_imp(score_ns, score_ew) for score_ns, score_ew in zip(scores_ns, scores_ew)]
    #
    # ed = time.perf_counter()
    # print(ed - st)
    # print(imps)
    # evaluator = Evaluator(50000, 8, "cuda")
    # random_net = PolicyNet()
    # for i in range(10):
    #     avg, sem, elapsed_time = evaluator.evaluate(sl_net(), random_net)
    #     print(avg, sem, elapsed_time)
    # dataset = load_rl_dataset("train")
    # deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    # vec_env = rl_cpp.ImpVecEnv()
    # replay_buffer = rl_cpp.ReplayBuffer(480, 38, 10000)
    # for i in range(10):
    #     env = rl_cpp.ImpEnv(deal_manager, [0, 0, 0, 0], replay_buffer, False)
    #     vec_env.push(env)
    # obs = {}
    # obs = vec_env.reset(obs)
    # print(obs)
    # with open("config/a2c.yaml") as f:
    #     cfg = yaml.safe_load(f)
    # print(cfg)
    # replay_buffer = rl_cpp.ReplayBuffer(10, 5, 100)
    # for i in range(5):
    #     replay_buffer.push(i * torch.ones(10), torch.tensor(i), torch.tensor(i), i * torch.ones(5))
    #
    # s = replay_buffer.dump()
    # print(s)
    # replay_buffer2 = rl_cpp.ReplayBuffer(10, 5, 100)
    # replay_buffer2.load(s)
    # print(replay_buffer2.dump())
    dataset = load_rl_dataset("vs_wb5_open_spiel")
    cards, ddts = dataset["cards"], dataset["ddts"]
    deal = rl_cpp.BridgeDeal()
    deal.cards = cards[0]
    deal.ddt = ddts[0]
    state = rl_cpp.BridgeBiddingState(deal)
    print(state)
