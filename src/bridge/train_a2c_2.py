"""
@file:train_policy_gradient_cpp
@author:qzz
@date:2023/3/1
@encoding:utf-8
"""
import argparse
import os
import time

from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from adan import Adan
from tqdm import trange

import rl_cpp
from src.bridge.agent_for_cpp import SingleEnvAgent
from src.bridge.bridge_actor import sl_net
from src.bridge.nets import PolicyNet
from src.bridge.bridge_vars import NUM_PLAYERS, PLUS_MINUS_SYMBOL, OBS_TENSOR_SIZE, NUM_CALLS
from src.bridge.utils import load_rl_dataset, Evaluator
from src.common_utils.array_utils import multiple_shuffle
from src.common_utils.logger import Logger
from src.common_utils.mem_utils import get_mem_usage
from src.common_utils.other_utils import set_random_seeds, mkdir_with_time
from src.common_utils.torch_utils import to_device
from src.common_utils.value_stats import MultiStats


# pylint: disable=unbalanced-tuple-unpacking
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
    parser.add_argument("--num_threads", type=int, default=7)
    parser.add_argument("--policy_lr", type=float, default=1e-5)
    parser.add_argument("--value_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--buffer_capacity", type=int, default=800000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--sample_batch_size", type=int, default=512)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--entropy_ratio", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--actor_update_freq", type=int, default=100)
    parser.add_argument("--opponent_update_freq", type=int, default=400)
    parser.add_argument("--save_dir", type=str, default="a2c")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_random_seeds(args.seed)
    device = args.device
    entropy_ratio = args.entropy_ratio
    clip_eps = args.clip_eps
    sample_batch_size = args.sample_batch_size

    supervised_net = sl_net(device=device)
    stats = MultiStats()
    save_dir: Optional[str] = None
    logger: Optional[Logger] = None
    if args.save_dir:
        save_dir = mkdir_with_time(args.save_dir)
        logger = Logger(os.path.join(save_dir, "log.txt"), auto_line_feed=True)

    checkpoint: [Optional[Dict]] = None
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        train_net = PolicyNet()
        train_net.load_state_dict(checkpoint["model_state_dict"]["policy"])
        train_net.to(device)
        print("Checkpoint loaded.")
    else:
        train_net = sl_net(device=device)
        print("No checkpoint file, use supervised learning network.")

    train_agent = SingleEnvAgent(train_net)
    train_agent.to(device)
    if checkpoint is not None:
        train_agent.v_net.load_state_dict(checkpoint["model_state_dict"]["value"])
    train_agent_jit = torch.jit.script(train_agent)
    oppo_agent = SingleEnvAgent(train_net)
    oppo_agent_jit = torch.jit.script(oppo_agent)
    train_locker = rl_cpp.ModelLocker([train_agent_jit], device)
    oppo_locker = rl_cpp.ModelLocker([oppo_agent_jit], device)

    # optimizer
    # p_opt = torch.optim.Adam(train_agent.p_net.parameters(), lr=args.policy_lr, eps=1e-8)
    # v_opt = torch.optim.Adam(train_agent.v_net.parameters(), lr=args.value_lr, eps=1e-8)
    p_opt = Adan(train_agent.p_net.parameters(), lr=args.policy_lr, eps=1e-8)
    v_opt = Adan(train_agent.v_net.parameters(), lr=args.value_lr, eps=1e-8)
    # p_opt = torch.optim.RMSprop(train_agent.p_net.parameters(), lr=args.policy_lr)
    # v_opt = torch.optim.RMSprop(train_agent.v_net.parameters(), lr=args.value_lr)
    if checkpoint is not None:
        p_opt.load_state_dict(checkpoint["optimizer_state_dict"]["policy"])
        print("policy optimizer state dict loaded.")
        if "value" in checkpoint["optimizer_state_dict"].keys():
            v_opt.load_state_dict(checkpoint["optimizer_state_dict"]["value"])
            print("value optimizer state dict loaded.")

    evaluator = Evaluator(10000, 8, "cuda")
    # evaluator = SingleProcessEvaluator(10000, args.eval_device)

    # random_net = random_policy_net()
    # for _ in range(5):
    #     st = time.perf_counter()
    #     avg, sem = evaluator.evaluate(train_agent.p_net, random_net)
    #     ed = time.perf_counter()
    #     print(avg, sem)
    #     print(ed - st)

    print("Creating training context....")
    st = time.perf_counter()
    IS_TRAIN_AGENT_GREEDY = 0
    IS_OPPO_AGENT_GREEDY = 0
    greedy_0 = [IS_TRAIN_AGENT_GREEDY, IS_OPPO_AGENT_GREEDY,
                IS_TRAIN_AGENT_GREEDY, IS_OPPO_AGENT_GREEDY]
    dataset = load_rl_dataset("train")
    cards, ddts = dataset["cards"], dataset["ddts"]
    replay_buffer = rl_cpp.ReplayBuffer(OBS_TENSOR_SIZE, NUM_CALLS, args.buffer_capacity)
    envs = []
    context = rl_cpp.Context()
    for i_thread in range(args.num_threads):
        cards_, ddts_ = multiple_shuffle(cards, ddts)
        env = rl_cpp.ImpEnv(cards_, ddts_, greedy_0, False)
        envs.append(env)
        actors = []
        for i in range(NUM_PLAYERS):
            actor = rl_cpp.SingleEnvActor(train_locker if i % 2 == 0 else oppo_locker, i, 1.0,
                                          False if i % 2 == 0 else True)
            actors.append(actor)
        t = rl_cpp.ImpEnvThreadLoop(actors, env, replay_buffer, False)
        context.push_thread_loop(t)

    ed = time.perf_counter()
    print(f"Elapsed time for creating context:{ed - st :.2f}")
    context.start()

    while replay_buffer.size() < args.burn_in_frames:
        print(f"\rwarming up replay buffer, {replay_buffer.size()}/{args.burn_in_frames}", end="")
        time.sleep(1)
    print()

    for epoch in range(args.num_epochs):
        mem_usage = get_mem_usage()
        print(f"Epoch {epoch}, mem usage is {mem_usage}")
        for batch_idx in trange(args.epoch_len):
            stats.feed("num_add", replay_buffer.num_add())
            num_update = batch_idx + epoch * args.epoch_len + 1

            p_opt.zero_grad()
            v_opt.zero_grad()
            batch_obs, batch_action, batch_reward, batch_log_probs \
                = replay_buffer.sample(sample_batch_size)
            batch_obs, batch_action, batch_reward, batch_log_probs \
                = to_device(batch_obs, batch_action, batch_reward,
                            batch_log_probs, device=args.device)
            p_loss, v_loss, entropy = train_agent.compute_a2c_loss_with_clip(batch_obs, batch_action,
                                                                             batch_reward, batch_log_probs,
                                                                             entropy_ratio, clip_eps)
            p_loss.backward()
            v_loss.backward()

            stats.feed("policy_loss", p_loss.item())
            stats.feed("value_loss", v_loss.item())
            stats.feed("entropy", entropy.item())
            torch.nn.utils.clip_grad_norm_(train_agent.p_net.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(train_agent.v_net.parameters(), args.max_grad_norm)
            p_opt.step()
            v_opt.step()

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
            "model_state_dict": {"policy": train_agent.p_net.state_dict(),
                                 "value": train_agent.v_net.state_dict()},
            "optimizer_state_dict": {"policy": p_opt.state_dict(),
                                     "value": v_opt.state_dict()},
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))

        msg = f"Epoch {epoch}, result is {avg}{PLUS_MINUS_SYMBOL}{sem}"
        logger.write(msg)
        context.resume()

    context.terminate()
