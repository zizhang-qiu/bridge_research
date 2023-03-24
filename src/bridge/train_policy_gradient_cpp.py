"""
@file:train_policy_gradient_cpp
@author:qzz
@date:2023/3/1
@encoding:utf-8
"""
import argparse
import copy
import logging
import os
import time

from typing import Optional, Tuple

import numpy as np
import torch
from adan import Adan
from matplotlib import pyplot as plt
from tqdm import trange

from src.bridge.agent_for_cpp import SingleEnvAgent, VecEnvAgent

import rl_cpp
from src.bridge.nets import PolicyNet
from src.bridge.bridge_vars import NUM_PLAYERS, PLUS_MINUS_SYMBOL
from src.bridge.utils import load_rl_dataset, sl_net, Evaluator
from src.common_utils.array_utils import get_avg_and_sem, multiple_shuffle
from src.common_utils.logger import Logger
from src.common_utils.mem_utils import get_mem_usage
from src.common_utils.other_utils import set_random_seeds, set_omp_threads, mkdir_with_time
from src.common_utils.saver import TopKSaver
from src.common_utils.torch_utils import to_device, clone_parameters
from src.common_utils.value_stats import MultiStats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--policy_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--buffer_capacity", type=int, default=800000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--sample_batch_size", type=int, default=512)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--entropy_ratio", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--actor_update_freq", type=int, default=50)
    parser.add_argument("--opponent_update_freq", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="../policy_gradient")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_random_seeds(args.seed)
    device = args.device
    entropy_ratio = args.entropy_ratio
    clip_eps = args.clip_eps
    sample_batch_size = args.sample_batch_size

    replay_buffer = rl_cpp.ReplayBuffer(519, 38, args.buffer_capacity)
    supervised_net = sl_net(device=device)
    stats = MultiStats()
    save_dir: Optional[str] = None
    logger: Optional[Logger] = None
    saver:Optional[TopKSaver] = None
    if args.save_dir:
        save_dir = mkdir_with_time(args.save_dir)
        logger = Logger(os.path.join(save_dir, "log.txt"), auto_line_feed=True)
        saver = TopKSaver(20, save_dir, "checkpoint")

    if args.checkpoint_path:
        train_net = PolicyNet()
        train_net.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"]["policy"])
        train_net.to(device)
        print("load trained net.")
    else:
        train_net = sl_net(device=device)
        print("no checkpoint, start from supervised learning net.")

    train_agent = SingleEnvAgent(train_net)
    train_agent.to(device)
    train_agent_jit = torch.jit.script(train_agent)
    oppo_agent = SingleEnvAgent(train_net)
    oppo_agent_jit = torch.jit.script(oppo_agent)
    train_locker = rl_cpp.ModelLocker([train_agent_jit], device)
    oppo_locker = rl_cpp.ModelLocker([oppo_agent_jit], device)
    # p_opt = torch.optim.Adam(train_agent.p_net.parameters(), lr=args.policy_lr, eps=1e-7)
    p_opt = Adan(train_agent.p_net.parameters(), lr=args.policy_lr)
    if args.checkpoint_path:
        p_opt.load_state_dict(torch.load(args.checkpoint_path)["optimizer_state_dict"]["policy"])
        print("load policy optimizer state dict")

    evaluator = Evaluator(10000, 8, "cuda")
    # random_net = random_policy_net()
    # for _ in range(5):
    #     st = time.perf_counter()
    #     avg, sem = evaluator.evaluate(train_agent.p_net, random_net)
    #     ed = time.perf_counter()
    #     print(avg, sem)
    #     print(ed - st)

    is_train_agent_greedy = False
    is_opponent_agent_greedy = False
    greedy_0 = [is_train_agent_greedy, is_opponent_agent_greedy, is_train_agent_greedy, is_opponent_agent_greedy]
    greedy_1 = [is_opponent_agent_greedy, is_train_agent_greedy, is_opponent_agent_greedy, is_train_agent_greedy]
    cards, ddts = load_rl_dataset("train", flatten=True)
    envs = []
    context = rl_cpp.Context()
    for i_thread in range(args.num_threads):
        cards_, ddts_ = multiple_shuffle(cards, ddts)
        env_0 = rl_cpp.ImpEnv(cards_, ddts_, greedy_0, False)
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
    while replay_buffer.size() < args.burn_in_frames:
        print(f"\rwarming up replay buffer, {replay_buffer.size()}/{args.burn_in_frames}", end="")
        time.sleep(1)
    print()

    logging.info("Start training...")

    for epoch in range(args.num_epochs):
        mem_usage = get_mem_usage()
        print(f"Epoch {epoch}, mem usage is {mem_usage}")
        for batch_idx in trange(args.epoch_len):
            stats.feed("num_add", replay_buffer.num_add())
            num_update = batch_idx + epoch * args.epoch_len + 1

            p_opt.zero_grad()
            batch_obs, batch_action, batch_reward, batch_log_probs = replay_buffer.sample(sample_batch_size)
            batch_obs, batch_action, batch_reward, batch_log_probs = \
                to_device(batch_obs, batch_action, batch_reward, batch_log_probs, device=args.device)
            loss = train_agent.compute_policy_gradient_loss(batch_obs, batch_action, batch_reward,
                                                            batch_log_probs, clip_eps, entropy_ratio)
            loss.backward()

            stats.feed("loss", loss.item())
            torch.nn.utils.clip_grad_norm_(train_agent.p_net.parameters(), args.max_grad_norm)
            p_opt.step()

            # check actor update
            if num_update % args.actor_update_freq == 0:
                train_agent_jit = torch.jit.script(train_agent)
                train_locker.update_model(train_agent_jit)

            # check opponent update
            if num_update % args.opponent_update_freq == 0:
                oppo_agent_jit = torch.jit.script(train_agent)
                oppo_locker.update_model(oppo_agent_jit)
                # time.sleep(0.1)
        # eval
        context.pause()
        while not context.all_paused():
            time.sleep(0.5)
        avg, sem = evaluator.evaluate(train_agent.p_net, supervised_net)
        stats.feed("avg_imp", avg)
        stats.feed("sem_imp", sem)
        num_deals_played_per_thread = [env.num_states() for env in envs]
        num_deals_played = sum(num_deals_played_per_thread)
        stats.feed("num_deals_played", num_deals_played)
        plt.bar(np.arange(len(num_deals_played_per_thread)), num_deals_played_per_thread)
        plt.savefig(os.path.join(save_dir, "num_deals_played_per_thread.png"))
        plt.close()
        # save
        stats.save_all(save_dir, True)
        checkpoint = {
            "model_state_dict": {"policy": copy.deepcopy(train_agent.p_net.state_dict())},
            "optimizer_state_dict": {"policy": copy.deepcopy(p_opt.state_dict())},
            "epoch": epoch
        }
        # torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))
        msg = f"Epoch {epoch}, result is {avg}{PLUS_MINUS_SYMBOL}{sem}"
        saver.save(checkpoint, avg)

        logger.write(msg)
        context.resume()

    context.terminate()
