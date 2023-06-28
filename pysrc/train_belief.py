import argparse
import copy
import os.path
from typing import Tuple

import numpy as np
import torch
from tqdm import trange

import rl_cpp
import utils
from nets import PolicyNet2
from agent_for_cpp import VecEnvAgent
from belief_model import BeliefModel
import common_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_net_path", type=str, default="a2c_fetch/4/folder_10/model2.pth")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_threads", type=int, default=6)
    parser.add_argument("--num_envs_per_thread", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--buffer_capacity", type=int, default=800000)
    parser.add_argument("--burn_in", type=int, default=200000)
    parser.add_argument("--save_dir", type=str, default="belief")
    return parser.parse_args()


def create_context(args: argparse.Namespace) -> Tuple[rl_cpp.Context, rl_cpp.ObsBeliefReplay]:
    net = PolicyNet2()
    net.load_state_dict(torch.load(args.policy_net_path)["model_state_dict"]["policy"])
    v_agent = VecEnvAgent(net).to(args.device)
    v_locker = rl_cpp.ModelLocker([torch.jit.script(copy.deepcopy(v_agent))], args.device)
    v_actor = rl_cpp.VecEnvActor(v_locker)
    # dataset = utils.load_rl_dataset("train")
    # deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    deal_manager = rl_cpp.RandomDealManager(1)
    context = rl_cpp.Context()
    replay = rl_cpp.ObsBeliefReplay(args.buffer_capacity, 1, 0.0, 0.0, 1)
    for i in range(args.num_threads):
        vec_env = rl_cpp.BridgeVecEnv()
        for j in range(args.num_envs_per_thread):
            env = rl_cpp.BridgeBiddingEnv(deal_manager, [1, 1, 1, 1])
            vec_env.push(env)
        t = rl_cpp.BeliefThreadLoop(v_actor, vec_env, replay)
        context.push_thread_loop(t)
    return context, replay


def train():
    torch.set_printoptions(threshold=100000)
    args = parse_args()
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    saver = common_utils.TopKSaver(save_dir, 10)
    stats = common_utils.MultiStats()
    belief_model = BeliefModel()
    belief_model.to("cuda")
    opt = torch.optim.Adam(lr=args.lr, params=belief_model.parameters(), fused=True)
    context, replay = create_context(args)
    print("Context created.")

    context.start()
    while (size := replay.size()) < args.burn_in:
        print(f"\rWarming up replay buffer: {size}/{args.burn_in}", end="")

    # Main loop
    for epoch in range(args.num_epochs):
        stats.reset()
        for batch_idx in trange(args.epoch_len):
            batch, weight = replay.sample(args.batch_size, args.device)
            # print(batch.belief)
            # print(batch.belief["belief"].sum(1))
            # input()
            loss = belief_model.compute_loss(batch)
            replay.update_priority(torch.Tensor())
            # print(loss)
            stats.feed("loss", loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        force_save_name = None
        if (epoch + 1) % 100 == 0:
            force_save_name = f"model_epoch{epoch}"
        saver.save(
            belief_model.state_dict(),
            -stats.get("loss").mean(),
            True,
            force_save_name=force_save_name,
        )
        print(f"Epoch {epoch}, mean loss: {stats.get('loss').mean()}, replay buffer:{replay.num_add()}")


if __name__ == '__main__':
    train()
