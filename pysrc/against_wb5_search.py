import os
import socket

import numpy as np
import torch
import rl_cpp
from agent_for_cpp import SingleEnvAgent, VecEnvAgent
from against_wb5 import controller_factory
from nets import PolicyNet2
from bluechip_bridge import BlueChipBridgeBot
from bridge_consts import NUM_PLAYERS
import common_utils
from pysrc.belief_model import BeliefModel
from utils import load_rl_dataset, tensor_dict_to_device

if __name__ == '__main__':
    device = "cuda"
    net = PolicyNet2()
    net.load_state_dict(torch.load("a2c_fetch/4/folder_10/model2.pth")["model_state_dict"]["policy"])
    net = net.to(device)
    agent = SingleEnvAgent(net).to(device)
    dataset = load_rl_dataset("vs_wb5_open_spiel")
    i_deal = 0
    imps = []
    bots = [BlueChipBridgeBot(i, controller_factory, 0) for i in range(NUM_PLAYERS)]
    v_agent = VecEnvAgent(net).to(device)
    v_locker = rl_cpp.ModelLocker([torch.jit.script(v_agent).to(device)], device)
    v_actor = rl_cpp.VecEnvActor(v_locker)
    belief_model = BeliefModel()
    belief_model.load_state_dict(torch.load("belief/folder_2/latest.pth"))
    b_locker = rl_cpp.ModelLocker([torch.jit.script(belief_model).to(device)], device)
    belief_model_cpp = rl_cpp.BeliefModel(b_locker)
    params = rl_cpp.SearchParams()
    params.verbose = True

    searcher = rl_cpp.Searcher(params, belief_model_cpp, v_actor)
    cards = dataset["cards"]
    ddts = dataset["ddts"]
    num_deals = cards.shape[0]
    save_dir = common_utils.mkdir_with_increment("vs_wbridge5")
    while i_deal < num_deals:
        try:
            deal = rl_cpp.BridgeDeal()
            deal.cards = cards[i_deal]
            deal.ddt = ddts[i_deal]
            state_0 = rl_cpp.BridgeBiddingState(deal)
            state_1 = rl_cpp.BridgeBiddingState(deal)

            for bot in bots:
                bot.restart()

            while not state_0.terminated():
                current_player = state_0.current_player()
                if current_player % 2 == 0:
                    obs = rl_cpp.make_obs_tensor_dict(state_0, 1)
                    obs = tensor_dict_to_device(obs, device)
                    reply = agent.simple_act(obs)
                    # action = reply["a"].item()
                    # print(action)
                    search_res = searcher.search(state_0, obs, reply)
                    print(search_res)
                    action = search_res["a"].item()
                else:
                    action = bots[current_player].step(state_0)
                # print(action)
                # print(action)
                state_0.apply_action(action)
                # print(state_0)

            # print(state_0)
            while not state_1.terminated():
                current_player = state_1.current_player()
                if current_player % 2 == 1:
                    obs = rl_cpp.make_obs_tensor_dict(state_1, 1)
                    obs = tensor_dict_to_device(obs, device)
                    reply = agent.simple_act(obs)
                    action = reply["a"].item()
                    # print(action)
                else:
                    action = bots[current_player].step(state_1)
                # print(action)
                state_1.apply_action(action)
                # print(state_1)
            # print(state_1)

            imp = rl_cpp.get_imp(int(state_0.returns()[0]), int(state_1.returns()[0]))
            imps.append(imp)
            msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\nimp: {imp}\n\n"
            with open(os.path.join(save_dir, f"log.txt"), "a") as f:
                f.write(msg)
            with open(os.path.join(save_dir, f"log_trajectory.txt"), "a") as f:
                f.write(f"open: {','.join(common_utils.to_str_list(state_0.history()))}\n"
                        f"close: {','.join(common_utils.to_str_list(state_1.history()))}\n")
            imps_np = np.array(imps)
            np.save(os.path.join(save_dir, f"imps.npy"), imps_np)
            i_deal += 1
            print(f"{i_deal}/{num_deals}, imp:{imp}")
        except socket.timeout as e:
            print(f"meet exception: {e}.")
            continue
    for bot in bots:
        bot.terminate()