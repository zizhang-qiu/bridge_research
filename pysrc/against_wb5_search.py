import os
import time
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing as mp

import common_utils
import rl_cpp
from against_wb5 import controller_factory
from agent_for_cpp import SingleEnvAgent, VecEnvAgent
from belief_model import BeliefModel
from bluechip_bridge import BlueChipBridgeBot
from bridge_consts import NUM_PLAYERS, PLUS_MINUS_SYMBOL
from nets import PolicyNet2
from utils import load_rl_dataset, tensor_dict_to_device


class Worker(mp.Process):
    def __init__(
        self,
        pid: int,
        cards: np.ndarray,
        ddts: np.ndarray,
        model_paths: Dict[str, str],
        device: str,
        save_dir: str,
    ):
        super().__init__()
        self.cards = cards
        self.ddts = ddts
        self.num_deals = cards.shape[0]
        self.model_paths = model_paths
        self.device = device
        self.save_dir = save_dir
        self.id = pid

    def run(self) -> None:
        net = PolicyNet2()
        net.load_state_dict(
            torch.load(self.model_paths["policy"])["model_state_dict"]["policy"]
        )
        agent = SingleEnvAgent(net).to(self.device)
        imps = []
        bots = [BlueChipBridgeBot(j, controller_factory, 0) for j in range(NUM_PLAYERS)]
        v_agent = VecEnvAgent(net).to(self.device)
        v_locker = rl_cpp.ModelLocker(
            [torch.jit.script(v_agent).to(self.device)], self.device
        )
        v_actor = rl_cpp.VecEnvActor(v_locker)
        belief_model = BeliefModel()
        belief_model.load_state_dict(torch.load(self.model_paths["belief"]))
        b_locker = rl_cpp.ModelLocker(
            [torch.jit.script(belief_model).to(self.device)], self.device
        )
        belief_model_cpp = rl_cpp.BeliefModel(b_locker)
        params = rl_cpp.SearchParams()
        params.verbose = False
        params.prob_exponent = 0.0
        params.length_exponent = 0
        params.max_prob = 1
        params.min_prob = 1e-2
        params.max_try = 1500
        params.max_particles = 100000
        params.temperature = 100
        params.max_rollouts = 1500
        params.min_rollouts = 100
        searcher = rl_cpp.Searcher(params, belief_model_cpp, v_actor)

        stats = common_utils.ValueStats()

        i_deal = 0
        while i_deal < self.num_deals:
            try:
                deal = rl_cpp.BridgeDeal()
                deal.cards = self.cards[i_deal]
                deal.ddt = self.ddts[i_deal]
                state_0 = rl_cpp.BridgeBiddingState(deal)
                state_1 = rl_cpp.BridgeBiddingState(deal)

                # for bot in bots:
                #     bot.restart()

                while not state_0.terminated():
                    current_player = state_0.current_player()
                    if current_player % 2 == 0:
                        obs = rl_cpp.make_obs_tensor_dict(state_0, 1)
                        st = time.perf_counter()
                        reply = agent.simple_act(
                            tensor_dict_to_device(obs, self.device)
                        )
                        search_res = searcher.search(state_0, obs, reply)
                        ed = time.perf_counter()
                        stats.feed(ed - st)
                        # print(search_res)
                        action = search_res["a"].item()
                    else:
                        action = bots[current_player].step(state_0)
                    # print(action)
                    # print(action)
                    state_0.apply_action(action)
                    # print(state_0)
                bots[0].restart()
                bots[2].restart()

                # print(state_0)
                while not state_1.terminated():
                    current_player = state_1.current_player()
                    if current_player % 2 == 1:
                        obs = rl_cpp.make_obs_tensor_dict(state_1, 1)
                        # obs = tensor_dict_to_device(obs, device)
                        st = time.perf_counter()
                        reply = agent.simple_act(
                            tensor_dict_to_device(obs, self.device)
                        )

                        search_res = searcher.search(state_1, obs, reply)
                        ed = time.perf_counter()
                        stats.feed(ed - st)
                        # print(search_res)
                        action = search_res["a"].item()
                        # print(action)
                    else:
                        action = bots[current_player].step(state_1)
                    # print(action)
                    state_1.apply_action(action)
                    # print(state_1)
                # print(state_1)
                bots[1].restart()
                bots[3].restart()

                imp = rl_cpp.get_imp(
                    int(state_0.returns()[0]), int(state_1.returns()[0])
                )
                imps.append(imp)
                msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\nimp: {imp}\n\n"
                with open(os.path.join(self.save_dir, f"log_{self.id}.txt"), "a") as f:
                    f.write(msg)
                with open(
                    os.path.join(self.save_dir, f"log_trajectory_{self.id}.txt"), "a"
                ) as f:
                    f.write(
                        f"open: {','.join(common_utils.to_str_list(state_0.history()))}\n"
                        f"close: {','.join(common_utils.to_str_list(state_1.history()))}\n"
                    )
                imps_np = np.array(imps)
                np.save(os.path.join(self.save_dir, f"imps_{self.id}.npy"), imps_np)
                i_deal += 1
                print(f"Process {self.id}, {i_deal}/{self.num_deals}, imp:{imp}")
            except Exception as e:
                print(f"Process {self.id} meet exception: {e}.")
                for bot in bots:
                    bot.restart()
                continue
        for bot in bots:
            bot.terminate()


if __name__ == "__main__":
    common_utils.set_random_seeds(32)
    common_utils.print_current_time()
    mp.set_start_method("spawn", force=True)
    device = "cuda"
    dataset = load_rl_dataset("vs_wb5_remained")
    num_process = 4
    start = 0
    num_deals = dataset["cards"].shape[0]
    print(num_deals)
    cards = dataset["cards"][start : start + num_deals]
    ddts = dataset["ddts"][start : start + num_deals]
    num_deals = cards.shape[0]
    num_deals_per_process = common_utils.allocate_tasks_uniformly(
        num_process, num_deals
    )
    indices_per_process = np.cumsum(np.insert(num_deals_per_process, 0, 0))
    # save_dir = common_utils.mkdir_with_increment("vs_wbridge5")
    # policy_path = "a2c_fetch/4/folder_10/model2.pth"
    # belief_path = "belief/folder_4/latest.pth"
    # model_paths = {
    #     "policy": policy_path,
    #     "belief": belief_path
    # }
    # workers = []
    # for i in range(num_process):
    #     left = indices_per_process[i]
    #     right = indices_per_process[i+1]
    #     # print(left, right)
    #     w = Worker(i, cards[left: right],
    #                ddts[left: right],
    #                model_paths, device, save_dir)
    #     workers.append(w)
    # for w in workers:
    #     w.start()
    #
    # for w in workers:
    #     w.join()
    #
    # # get average and standard error of mean
    # imps_list = [np.load(os.path.join(save_dir, f"imps_{i}.npy")) for i in range(num_process)]
    # final_imps = np.concatenate(imps_list)
    # avg, sem = common_utils.get_avg_and_sem(final_imps)
    # msg = f"result is {avg}{PLUS_MINUS_SYMBOL}{sem}"
    # print(msg)
    # np.save(os.path.join(save_dir, "imps.npy"), final_imps)
