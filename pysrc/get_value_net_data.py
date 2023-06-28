import copy
import pickle
import time
from typing import List

import numpy as np
import torch
from tqdm import trange

from nets import PolicyNet, PolicyNet2
from agent_for_cpp import VecEnvAgent
import rl_cpp
from pysrc.utils import load_rl_dataset


def analyze(net: PolicyNet2, cards: np.ndarray, ddts: np.ndarray, par_scores: np.ndarray, device: str = "cuda") \
        -> List[rl_cpp.BridgeBiddingEnv]:
    agent = VecEnvAgent(copy.deepcopy(net)).to(device)
    model_locker = rl_cpp.ModelLocker([torch.jit.script(agent).to(device)], device)
    actor = rl_cpp.VecEnvActor(model_locker)
    num_deals = cards.shape[0]
    num_envs_per_thread = num_deals // 8
    vec_envs = []
    ctx = rl_cpp.Context()
    for i in range(8):
        vec_env = rl_cpp.BridgeVecEnv()
        left = i * num_envs_per_thread
        right = left + num_envs_per_thread
        manager = rl_cpp.BridgeDealManager(cards[left: right], ddts[left:right], par_scores[left:right])
        for j in range(num_envs_per_thread):
            env = rl_cpp.BridgeBiddingEnv(manager, [1, 1, 1, 1])
            vec_env.push(env)
        vec_envs.append(vec_env)
        t = rl_cpp.VecEnvAllTerminateThreadLoop(actor, vec_env)
        ctx.push_thread_loop(t)
    ctx.start()
    while not ctx.terminated():
        time.sleep(1)
    envs = []
    for vec_env in vec_envs:
        envs.extend(vec_env.get_envs())
    return envs


if __name__ == '__main__':
    net = PolicyNet2()
    net.load_state_dict(torch.load("a2c_fetch/4/folder_10/model2.pth")["model_state_dict"]["policy"])
    dataset = load_rl_dataset("train")
    max_size = 250000
    num_batches = dataset["cards"].shape[0] // max_size
    all_envs = []
    for i in trange(num_batches):
        # print(i)
        cards = dataset["cards"][i * max_size: (i + 1) * max_size]
        ddts = dataset["ddts"][i * max_size: (i + 1) * max_size]
        par_scores = dataset["par_scores"][i * max_size: (i + 1) * max_size]
        envs = analyze(net, cards, ddts, par_scores)
        all_envs.extend(envs)
    histories = []
    for env in all_envs:
        history = env.get_state().history()
        histories.append(history)
    print(len(histories))
    with open("dataset/value/trajectories.pkl", "wb") as fp:
        pickle.dump(histories, fp)
