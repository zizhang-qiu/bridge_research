import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import rl_cpp
from belief_model import BeliefModel
from agent_for_cpp import VecEnvAgent
from nets import PolicyNet2
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--belief_path", type=str, default="belief/folder_4/latest.pth")
    parser.add_argument("--policy_path", type=str, default="a2c_fetch/4/folder_10/model2.pth")
    parser.add_argument("--save_dir", type=str, default="belief_quality")
    parser.add_argument("--buffer_capacity", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    belief_net = BeliefModel()
    belief_net.load_state_dict(torch.load(args.belief_path))
    belief_net.to("cuda")

    policy_net = PolicyNet2()
    # print(torch.load(args.policy_path))
    policy_net.load_state_dict(torch.load(args.policy_path)["model_state_dict"]["policy"])
    policy_net.to("cuda")
    agent = VecEnvAgent(policy_net)
    model_locker = rl_cpp.ModelLocker([torch.jit.script(agent).to("cuda")], "cuda")
    actor = rl_cpp.VecEnvActor(model_locker)
    deal_manager = rl_cpp.RandomDealManager(1)
    replay = rl_cpp.ObsBeliefReplay(args.buffer_capacity, 1, 0.0, 0.0, 1)
    vec_env = rl_cpp.BridgeVecEnv()
    for j in range(args.num_games):
        env = rl_cpp.BridgeBiddingEnv(deal_manager, [0, 0, 0, 0])
        vec_env.push(env)
    t = rl_cpp.BeliefThreadLoop(actor, vec_env, replay)
    t.main_loop()
    print(replay.size())
    transitions, _ = replay.sample(replay.size(), "cuda")
    # print(transitions.obs)
    # print(transitions.belief)
    # print(transitions.length)
    # print(transitions.length.max())
    print(torch.unique(transitions.length))
    print(torch.bincount(transitions.length))

    belief_probs = belief_net.get_belief(transitions)
    # loss = F.binary_cross_entropy(belief_probs, transitions.belief["belief"], reduction="none")
    loss = -transitions.belief["belief"] * torch.log(belief_probs + 1e-15)
    # print(loss)
    # print(loss.shape)

    losses_per_length: List[torch.Tensor] = []
    for length in torch.unique(transitions.length):
        idx = torch.where(transitions.length == length)[0]
        # print(idx)
        losses_per_length.append(loss[idx])
    # print(losses_per_length)
    # for item in losses_per_length:
    #     print(item.shape)

    avg_losses = [loss_.sum(1).mean().item() for loss_ in losses_per_length]
    print(avg_losses)

    # plt.plot(torch.unique(transitions.length).cpu(), avg_losses)
    # plt.show()
    np.save("belief_loss.npy", np.array(avg_losses))







if __name__ == '__main__':
    main()
