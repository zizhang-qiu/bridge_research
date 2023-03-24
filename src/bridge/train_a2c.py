"""
@file:train_policy_gradient_cpp
@author:qzz
@date:2023/3/1
@encoding:utf-8
"""
import argparse
import logging
import os
import time

from typing import Optional, Tuple, Dict

import torch
from tqdm import trange

from src.bridge.agent_for_cpp import SingleEnvAgent, VecEnvAgent

import rl_cpp
from src.bridge.bridge_actor import sl_net, random_policy_net
from src.bridge.nets import PolicyNet
from src.bridge.bridge_vars import NUM_PLAYERS, PLUS_MINUS_SYMBOL
from src.bridge.utils import load_rl_dataset
from src.common_utils.array_utils import get_avg_and_sem
from src.common_utils.logger import Logger
from src.common_utils.mem_utils import get_mem_usage
from src.common_utils.other_utils import set_random_seeds,mkdir_with_time
from src.common_utils.torch_utils import TopkSaver
from src.common_utils.value_stats import MultiStats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--num_threads", type=int, default=7)
    parser.add_argument("--policy_lr", type=float, default=1e-6)
    parser.add_argument("--value_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--burn_in_frames", type=int, default=40000)
    parser.add_argument("--buffer_capacity", type=int, default=400000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--epoch_len", type=int, default=2000)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--entropy_ratio", type=float, default=0.01)
    parser.add_argument("--actor_update_freq", type=int, default=50)
    parser.add_argument("--opponent_update_freq", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="a2c")
    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, num_deals: int, eval_device: str):
        _cards, _ddts = load_rl_dataset("valid", flatten=True)
        self.cards = _cards[:num_deals]
        self.ddts = _ddts[:num_deals]
        self.device = eval_device
        self.num_deals = num_deals
        # when evaluating, all actors choose greedy action
        self.greedy = [1, 1, 1, 1]

    def evaluate(self, _train_net: PolicyNet, _oppo_net: PolicyNet) -> Tuple[float, float]:
        # print("Enter evaluate.")
        _train_net = _train_net.to(self.device)
        _oppo_net = _oppo_net.to(self.device)
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
        for i_env in range(self.num_deals):
            _left = i_env
            _right = _left + 1
            _env_0 = rl_cpp.BridgeBiddingEnv(self.cards[_left:_right],
                                             self.ddts[_left:_right],
                                             self.greedy)
            _env_1 = rl_cpp.BridgeBiddingEnv(self.cards[_left:_right],
                                             self.ddts[_left:_right],
                                             self.greedy)
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
        torch.cuda.empty_cache()
        return get_avg_and_sem(imps)


if __name__ == '__main__':
    args = parse_args()
    set_random_seeds(args.seed)
    device = args.device
    entropy_ratio = args.entropy_ratio
    sample_batch_size = args.sample_batch_size

    replay_buffer = rl_cpp.ReplayBuffer(519, args.buffer_capacity)
    supervised_net = sl_net(device=device)
    stats = MultiStats()
    save_dir: Optional[str] = None
    logger: Optional[Logger] = None
    saver:Optional[TopkSaver] = None
    if args.save_dir:
        save_dir = mkdir_with_time(args.save_dir)
        logger = Logger(os.path.join(save_dir, "log.txt"), auto_line_feed=True)
        saver = TopkSaver(save_dir, 10)

    checkpoint: [Optional[Dict]] = None
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        train_net = PolicyNet()
        train_net.load_state_dict(checkpoint["model_state_dict"]["policy"])
        train_net.to(device)
        print("checkpoint loaded.")
    else:
        train_net = sl_net(device=device)
        print("no checkpoint file, use supervised learning network.")

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
    p_opt = torch.optim.Adam(train_agent.p_net.parameters(), lr=args.policy_lr, eps=1e-7)
    v_opt = torch.optim.Adam(train_agent.v_net.parameters(), lr=args.value_lr, eps=1e-7)
    if checkpoint is not None:
        p_opt.load_state_dict(checkpoint["optimizer_state_dict"]["policy"])
        print("policy optimizer state dict loaded.")
        if "value" in checkpoint["optimizer_state_dict"].keys():
            v_opt.load_state_dict(checkpoint["optimizer_state_dict"]["value"])
            print("value optimizer state dict loaded.")

    random_net = random_policy_net(device=device)
    evaluator = Evaluator(10000, "cuda")

    is_train_agent_greedy = False
    is_opponent_agent_greedy = False
    greedy_0 = [is_train_agent_greedy, is_opponent_agent_greedy, is_train_agent_greedy, is_opponent_agent_greedy]
    greedy_1 = [is_opponent_agent_greedy, is_train_agent_greedy, is_opponent_agent_greedy, is_train_agent_greedy]
    cards, ddts = load_rl_dataset("train", flatten=True)
    seg = len(ddts) // args.num_threads
    context = rl_cpp.Context()
    envs = []
    for i_thread in range(args.num_threads):
        left = i_thread * seg
        right = left + seg
        env_0 = rl_cpp.BridgeBiddingEnv(cards[left:right], ddts[left:right],
                                        greedy_0)
        env_1 = rl_cpp.BridgeBiddingEnv(cards[left:right], ddts[left:right],
                                        greedy_1)
        envs.append(env_0)
        actors = []
        for i in range(NUM_PLAYERS):
            actor = rl_cpp.SingleEnvActor(train_locker if i % 2 == 0 else oppo_locker, i, 1.0,
                                          False if i % 2 == 0 else True)
            actors.append(actor)
        t = rl_cpp.BridgePGThreadLoop(actors,
                                      env_0,
                                      env_1,
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
            num_deals_played = sum([env.num_states() for env in envs])
            stats.feed("num_deals_played", num_deals_played)
            stats.feed("num_add", replay_buffer.num_add())
            num_update = batch_idx + epoch * args.epoch_len + 1

            p_opt.zero_grad()
            v_opt.zero_grad()
            batch_obs, batch_action, batch_reward = replay_buffer.sample(sample_batch_size)
            p_loss, v_loss = train_agent.compute_a2c_loss(batch_obs, batch_action, batch_reward, entropy_ratio)
            p_loss.backward()
            v_loss.backward()

            stats.feed("policy_loss", p_loss.item())
            stats.feed("value_loss", v_loss.item())
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

        # save
        stats.save_all(save_dir, True)
        checkpoint = {
            "model_state_dict": {"policy": train_agent.p_net.state_dict(), "value": train_agent.v_net.state_dict()},
            "optimizer_state_dict": {"policy": p_opt.state_dict(), "value": v_opt.state_dict()},
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))
        msg = f"Epoch {epoch}, result is {avg}{PLUS_MINUS_SYMBOL}{sem}"
        logger.write(msg)
        context.resume()

    context.terminate()
