"""
@file:threads_benchmark
@author:qzz
@date:2023/4/4
@encoding:utf-8
"""
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

import rl_cpp
from agent_for_cpp import SingleEnvAgent
from bridge_vars import NUM_PLAYERS
from utils import sl_net, load_rl_dataset, Evaluator
from common_utils.array_utils import multiple_shuffle

device = "cuda"


def benchmark(num_threads: int, buffer_capacity: int):
    net1 = sl_net()
    net2 = sl_net()
    train_agent = SingleEnvAgent(net1)
    train_agent.to(device)
    train_agent_jit = torch.jit.script(train_agent)
    oppo_agent = SingleEnvAgent(net2)
    oppo_agent_jit = torch.jit.script(oppo_agent)
    train_locker = rl_cpp.ModelLocker([train_agent_jit], device)
    oppo_locker = rl_cpp.ModelLocker([oppo_agent_jit], device)
    dataset = load_rl_dataset("train")
    cards, ddts = dataset["cards"], dataset["ddts"]
    envs = []
    replay_buffer = rl_cpp.ReplayBuffer(519, 38, buffer_capacity)
    context = rl_cpp.Context()
    for i_thread in range(num_threads):
        cards_, ddts_ = multiple_shuffle(cards, ddts)
        env_0 = rl_cpp.ImpEnv(cards_, ddts_, [1, 1, 1, 1], False)
        envs.append(env_0)

        actors = []
        for i in range(NUM_PLAYERS):
            actor = rl_cpp.SingleEnvActor(train_locker if i % 2 == 0 else oppo_locker, i, 1.0,
                                          False if i % 2 == 0 else True)
            actors.append(actor)
        t = rl_cpp.ImpEnvThreadLoop(actors,
                                    env_0,
                                    replay_buffer,
                                    False)
        context.push_thread_loop(t)
    context.start()

    # burn in
    st = time.perf_counter()
    while replay_buffer.size() < buffer_capacity:
        print(f"\rWarming up replay buffer, {replay_buffer.size()}/{buffer_capacity}", end="")
        time.sleep(1)
    print()
    ed = time.perf_counter()
    return ed - st


def main():
    capacity = 100000
    elapsed_times = np.zeros(8, dtype=float)
    for i_thread in range(1, 9):
        elapsed_time = benchmark(i_thread, capacity)
        print(f"{i_thread} threads, elapsed time: {elapsed_time}.")
        elapsed_times[i_thread - 1] = elapsed_time

    np.save("threads_benchmark.npy", elapsed_times)
    plt.bar(np.arange(1, 9), elapsed_times)
    plt.savefig("threads_benchmark.png")
    plt.close()


def eval_benchmark():
    num_deals = 50000
    net1 = sl_net(device="cuda")
    net2 = sl_net(device="cuda")
    for num_threads in [1, 2, 4, 5, 8]:
        evaluator = Evaluator(num_deals, num_threads, "cuda")
        avg, sem, elapsed_time, _, _ = evaluator.evaluate(net1, net2)
        print(f"{num_threads} threads, eval time is {elapsed_time:.2f}")


if __name__ == '__main__':
    eval_benchmark()
