import pickle
from typing import List

import numpy as np
import torch

import belief_model
import common_utils
import rl_cpp
from against_wb5 import controller_factory
from agent_for_cpp import SingleEnvAgent, VecEnvAgent
from bluechip_bridge import BlueChipBridgeBot
from bridge_consts import NUM_PLAYERS, PLUS_MINUS_SYMBOL
from imp_result import IMPResult, get_imp_list
from nets import PolicyNet2
from utils import load_rl_dataset, tensor_dict_to_device


def run_once(card, ddt, device, bots):
    deal = rl_cpp.BridgeDeal()
    deal.cards = card
    deal.ddt = ddt
    state_0 = rl_cpp.BridgeBiddingState(deal)
    state_1 = rl_cpp.BridgeBiddingState(deal)
    print(state_0)

    for bot in bots:
        bot.restart()

    while not state_0.terminated():
        current_player = state_0.current_player()
        if current_player % 2 == 0:
            obs = rl_cpp.make_obs_tensor_dict(state_0, 1)
            reply = agent.simple_act(tensor_dict_to_device(obs, device))
            # action = reply["a"].item()
            # print(action)
            search_res = searcher.search(state_0, obs, reply)
            # print(search_res)
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
            # obs = tensor_dict_to_device(obs, device)
            reply = agent.simple_act(tensor_dict_to_device(obs, device))
            search_res = searcher.search(state_1, obs, reply)
            # print(search_res)
            action = search_res["a"].item()
            # print(action)
        else:
            action = bots[current_player].step(state_1)
        # print(action)
        state_1.apply_action(action)
        # print(state_1)
    # print(state_1)
    print(state_0)
    print(state_1)

    imp = rl_cpp.get_imp(int(state_0.returns()[0]), int(state_1.returns()[0]))
    msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\n"
    return imp, msg, state_0, state_1


if __name__ == "__main__":
    common_utils.set_random_seeds(42)
    original_imp = np.load("vs_wbridge5/folder_58/imps.npy")
    imp_file_path = "vs_wbridge5/imps.npy"
    imps = np.load(imp_file_path)
    # print(imps)
    device = "cuda"
    dataset = load_rl_dataset("vs_wb5_open_spiel")
    policy_path = "a2c_fetch/4/folder_10/model2.pth"
    belief_path = "belief/folder_4/latest.pth"
    net = PolicyNet2()
    net.load_state_dict(torch.load(policy_path)["model_state_dict"]["policy"])
    agent = SingleEnvAgent(net).to(device)

    bots = [BlueChipBridgeBot(j, controller_factory, 0) for j in range(NUM_PLAYERS)]
    v_agent = VecEnvAgent(net).to(device)
    v_locker = rl_cpp.ModelLocker([torch.jit.script(v_agent).to(device)], device)
    v_actor = rl_cpp.VecEnvActor(v_locker)
    belief_model = belief_model.BeliefModel()
    belief_model.load_state_dict(torch.load(belief_path))
    b_locker = rl_cpp.ModelLocker([torch.jit.script(belief_model).to(device)], device)
    belief_model_cpp = rl_cpp.BeliefModel(b_locker)
    params = rl_cpp.SearchParams()
    params.verbose = False
    params.random_sample = False
    params.prob_exponent = 0
    params.length_exponent = 0
    params.max_prob = 1
    params.min_prob = 1e-2
    params.max_try = 1500
    params.temperature = 100
    searcher = rl_cpp.Searcher(params, belief_model_cpp, v_actor)
    cards = dataset["cards"]
    ddts = dataset["ddts"]
    # num_trial = 3
    start = 0
    with open("imp_results_rl_bmcs.pkl", "rb") as fp:
        current_result: List[IMPResult] = pickle.load(fp)
    imp_list = get_imp_list(current_result)
    # print(imp_list)
    print(np.mean(imp_list))
    imp_array = np.array(imp_list, dtype=np.float32)
    sample_weight = np.abs(np.clip(imp_array, -24, 0))
    sample_weight /= np.sum(sample_weight)
    np.set_printoptions(threshold=1000000)
    # print(sample_weight)
    while True:
        index = np.random.choice(np.arange(len(sample_weight)), p=sample_weight)
        print(f"Running deal No.{index}, original imp:{imp_array[index]}.")
        num_trial = 0
        current_cards = current_result[index].cards
        current_ddt = current_result[index].ddt
        while num_trial < 1:
            try:
                imp, state_str, open_state, close_state = run_once(current_cards, current_ddt, device, bots)
                print(f"Trial {num_trial}: {imp}, original:{imp_array[index]}.\n{state_str}")
                num_trial += 1
            except Exception:
                continue
            if imp > current_result[index].imp:
                original_imp = current_result[index].imp
                current_result[index] = IMPResult(current_cards, current_ddt, open_state, close_state)
                imp_array[index] = imp
                avg, sem = common_utils.get_avg_and_sem(imp_array)
                print(f"Deal No.{index} improved, {original_imp}->{imp}, "
                      f"final result:{avg}{PLUS_MINUS_SYMBOL}{sem}")
                with open("imp_results_rl_bmcs.pkl", "wb") as fp:
                    pickle.dump(current_result, fp)

                # update weight
                sample_weight = np.abs(np.clip(imp_array, -24, 0))
                sample_weight /= np.sum(sample_weight)
    # with open("vs_wbridge5/log.pkl", "rb") as fp:
    #     state_strs = pickle.load(fp)
    # for i, s in enumerate(state_strs):
    #     if s == "":
    #         start = i
    #         break
    # print(f"start:{start}")
    # for index in range(start, 500):
    #     best_imp = -100
    #     best_state_str = ""
    #     print(f"Deal {index}., original imp: {imps[index]}")
    #     num_trial = 0
    #     while num_trial < 3:
    #         try:
    #             imp, state_str = run_once(cards[index], ddts[index], device, bots)
    #             print(f"Trial {num_trial}: {imp}, original:{imps[index]}.")
    #             num_trial += 1
    #         except Exception:
    #             continue
    #         if imp > best_imp:
    #             best_imp = imp
    #             best_state_str = state_str
    #         if imp >= original_imp[index]:
    #             break
    #     imps[index] = best_imp
    #     state_strs[index] = best_state_str
    #     np.save(imp_file_path, imps)
    # with open("vs_wbridge5/log.pkl", "wb") as fp:
    #     pickle.dump(state_strs, fp)
