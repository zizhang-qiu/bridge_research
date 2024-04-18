import argparse
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import rl_cpp
from belief_model import BeliefModel
from agent_for_cpp import VecEnvAgent
from nets import PolicyNet2
from utils import tensor_dict_to_device
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_belief_per_length", type=int, default=1000)
    parser.add_argument("--num_total_nodes", type=int, default=100000)
    parser.add_argument("--belief_path", type=str, default="belief/folder_4/latest.pth")
    parser.add_argument(
        "--policy_path", type=str, default="a2c_fetch/4/folder_10/model2.pth"
    )
    parser.add_argument("--num_random_sample", type=int, default=1000)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="belief_quality")
    return parser.parse_args()


def get_random_belief(env: rl_cpp.BridgeBiddingEnv, num_sample: int):
    player = env.current_player()
    state = env.get_state()
    cards = state.get_player_cards(player)

    remained_cards = set(range(52)) - set(cards)
    remained_cards = list(remained_cards)
    for i in range(num_sample):
        random.shuffle(remained_cards)


def main():
    args = parse_args()
    print(vars(args))
    belief_net = BeliefModel()
    belief_net.load_state_dict(torch.load(args.belief_path))
    belief_net.to("cuda")

    policy_net = PolicyNet2()
    # print(torch.load(args.policy_path))
    policy_net.load_state_dict(
        torch.load(args.policy_path)["model_state_dict"]["policy"]
    )
    policy_net.to("cuda")

    agent = VecEnvAgent(policy_net).to("cuda")
    deal_manager = rl_cpp.RandomDealManager(1)

    env = rl_cpp.BridgeBiddingEnv(deal_manager, [0, 0, 0, 0])

    num_lengths = args.max_length - args.min_length + 1
    length_counter = torch.zeros(num_lengths, dtype=torch.int32)
    node_per_length = [[] for _ in range(num_lengths)]
    belief_per_length = [[] for _ in range(num_lengths)]
    random_belief_losses = [[] for _ in range(num_lengths)]
    # print(length_counter.shape)
    while not torch.all(length_counter == args.num_belief_per_length):
        if env.terminated():
            env.reset()
        # print(env)
        current_player = env.current_player()
        obs = env.get_feature()
        belief = env.get_state().hidden_observation_tensor()
        # print(obs)
        length = len(env.get_state().bid_history())

        reply = agent.act(tensor_dict_to_device(obs, "cuda"))
        # print(reply)
        env.step(reply)
        if (
            length < args.min_length
            or length > args.max_length
            or length_counter[length - args.min_length] == args.num_belief_per_length
        ):
            continue

        node_per_length[length - args.min_length].append(obs)
        belief_per_length[length - args.min_length].append(torch.tensor(belief))

        random_beliefs = []
        for i in range(args.num_random_sample):
            cards = rl_cpp.random_sample(env.get_state(), current_player)
            deal = rl_cpp.BridgeDeal()
            deal.cards = cards
            state = rl_cpp.BridgeBiddingState(deal)
            for action in env.get_state().bid_history()[:-1]:
                state.apply_action(action)
            random_belief = state.hidden_observation_tensor()
            random_beliefs.append(torch.tensor(random_belief))
        random_beliefs = torch.stack(random_beliefs, 0)
        # belief_pred = belief_net(obs["s"].to("cuda"))
        random_belief_loss = (
            -torch.log(random_beliefs + 1e-15) * torch.tensor(belief).cpu()
        )
        # random_belief_loss = F.cross_entropy(random_beliefs, torch.tensor(belief))
        random_belief_losses[length - args.min_length].append(
            random_belief_loss.sum(1).mean().item()
        )

        length_counter[length - args.min_length] += 1

        # print(length_counter)
        print(sum(length_counter))
        if sum(length_counter) >= args.num_total_nodes:
            break

    losses_per_length = []
    for i in range(num_lengths):
        if length_counter[i] > 0:
            belief_pred = belief_net(
                rl_cpp.tensor_dict_stack(node_per_length[i], 0)["s"].to("cuda")
            )
            loss = -torch.log(belief_pred.cpu() + 1e-15) * torch.stack(
                belief_per_length[i], 0
            )
            losses_per_length.append(loss.sum(1).mean().item())
        else:
            losses_per_length.append(0)
    print(losses_per_length)
    random_losses = [np.mean(ls) for ls in random_belief_losses]
    print(random_losses)

    plt.figure()
    plt.plot(range(len(losses_per_length)), losses_per_length, label="belief")
    plt.plot(range(len(losses_per_length)), random_losses, label="random")
    plt.xlabel("Bidding length")
    plt.ylabel("Cross entropy")
    plt.xticks(np.arange(21))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
