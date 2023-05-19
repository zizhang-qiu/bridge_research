"""
@file:against_wb5
@author:qzz
@date:2023/3/4
@encoding:utf-8
"""
import os
import pprint
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

import common_utils
import rl_cpp
from against_wb5_search import _WBridge5Client
from agent_for_cpp import SingleEnvAgent
from bluechip_bridge import BlueChipBridgeBot
from bridge_vars import NUM_PLAYERS, PLUS_MINUS_SYMBOL
from nets import PolicyNet
from utils import load_rl_dataset, tensor_dict_to_device, sl_net


def controller_factory(port: int):
    """Implements bluechip_bridge.BlueChipBridgeBot."""
    client = _WBridge5Client("d:/wbridge5/Wbridge5.exe Autoconnect {port}", port=port)
    client.start()
    return client


class AgainstWb5Worker(mp.Process):
    def __init__(self, process_id: int, num_deals: int, port: int, cards: np.ndarray, ddts: np.ndarray,
                 shared_dict: Dict, device: str, save_dir: Optional[str] = None):
        """
        A worker to play against Wbridge5.
        Args:
            process_id: The id of the process
            num_deals: The number of deals to play
            port: The port number to connect with Wbridge5
            cards: The cards array.
            ddts: The ddts array.
            shared_dict: The shared dict of policy network.
            device: The device of the agent.
            save_dir: The directory to save results.
        """
        super().__init__()
        common_utils.assert_eq(num_deals, cards.shape[0])
        common_utils.assert_eq(num_deals, ddts.shape[0])
        self._process_id = process_id
        self._num_deals = num_deals
        self._cards = cards
        self._ddts = ddts
        self._port = port
        self._device = device
        self._shared_dict = shared_dict.copy()
        self.logger: Optional[common_utils.Logger] = None
        self.save_dir = save_dir

    def run(self) -> None:
        net = PolicyNet()
        net.load_state_dict(self._shared_dict)
        net.to(self._device)
        agent = SingleEnvAgent(net).to(self._device)
        i_deal = 0
        _imps = []
        bots = [BlueChipBridgeBot(i, controller_factory, self._port + i) for i in range(NUM_PLAYERS)]
        while i_deal < self._num_deals:
            try:
                deal = rl_cpp.BridgeDeal()
                deal.cards = self._cards[i_deal]
                deal.ddt = self._ddts[i_deal]
                state_0 = rl_cpp.BridgeBiddingState(deal)
                state_1 = rl_cpp.BridgeBiddingState(deal)
                for bot in bots:
                    bot.restart()

                while not state_0.terminated():
                    current_player = state_0.current_player()
                    if current_player % 2 == 0:
                        obs = rl_cpp.make_obs_tensor_dict(state_0, 1)
                        obs = tensor_dict_to_device(obs, self._device)
                        reply = agent.simple_act(obs)
                        action = reply["a"].item()
                        # print(action)
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
                        obs = tensor_dict_to_device(obs, self._device)
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
                _imps.append(imp)
                msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\nimp: {imp}\n\n"
                with open(os.path.join(self.save_dir, f"log_{self._process_id}.txt"), "a") as f:
                    f.write(msg)
                with open(os.path.join(self.save_dir, f"log_{self._process_id}_trajectory.txt"), "a") as f:
                    f.write(f"open: {','.join(common_utils.to_str_list(state_0.history()))}\n"
                            f"close: {','.join(common_utils.to_str_list(state_1.history()))}\n")
                _imps_np = np.array(_imps)
                np.save(os.path.join(self.save_dir, f"imps_{self._process_id}.npy"), _imps_np)
                i_deal += 1
                print(f"Process {self._process_id}, {i_deal}/{self._num_deals}, imp:{imp}")
            except Exception as e:
                print(f"Process {self._process_id} meet exception: {e}.")
                continue
        for bot in bots:
            bot.terminate()


def main():
    # cards, ddts = load_rl_dataset("vs_wb5_fb", flatten=True)
    dataset = load_rl_dataset("vs_wb5_open_spiel")
    cards = dataset["cards"]
    ddts = dataset["ddts"]
    with open("conf/against_wb5.yaml", "r") as fp:
        config: Dict = yaml.safe_load(fp)
    net = PolicyNet()
    net.load_state_dict(torch.load(config["checkpoint_path"])["model_state_dict"]["policy"])
    # net.load_state_dict(torch.load(config["checkpoint_path"])["model_state_dict"])
    net.to("cuda")
    num_processes = config["num_processes"]
    num_deals_per_process = common_utils.allocate_tasks_uniformly(num_processes, config["num_deals"])
    shared_dict = common_utils.create_shared_dict(net)
    save_dir = common_utils.mkdir_with_increment(config["save_dir"])
    logger = common_utils.Logger(os.path.join(save_dir, "log.txt"), auto_line_feed=True, verbose=False)
    logger.write(pprint.pformat(config))
    base_port = 7000
    start = config["start"]
    st = time.perf_counter()
    # w = AgainstWb5Worker(0, 550, base_port, cards[start:start + 550], ddts[start:start + 550], shared_dict, "cuda",
    #                      save_dir)
    # w.run()
    # print(time.perf_counter() - st)

    workers = []
    for i in range(num_processes):
        num_deals = num_deals_per_process[i]
        w = AgainstWb5Worker(i, num_deals,
                             base_port + 20 * i,
                             cards[start + num_deals * i:start + num_deals * (i + 1)],
                             ddts[start + num_deals * i:start + num_deals * (i + 1)],
                             shared_dict, "cuda", save_dir)
        workers.append(w)
    for w in workers:
        w.start()

    for w in workers:
        w.join()

    # get average and standard error of mean
    imps_list = [np.load(os.path.join(save_dir, f"imps_{i}.npy")) for i in range(num_processes)]
    imps = np.concatenate(imps_list)
    avg, sem = common_utils.get_avg_and_sem(imps)
    msg = f"result is {avg}{PLUS_MINUS_SYMBOL}{sem}"
    print(msg)
    logger.write(msg)
    np.save(os.path.join(save_dir, "imps.npy"), imps)
    print(f"The whole process takes {time.perf_counter() - st:.2f} seconds.")

    # gather logs
    logs = []
    for i in range(num_processes):
        with open(os.path.join(save_dir, f"log_{i}.txt"), "r") as f:
            content = f.read()
        logs.append(content)

    logger.write("\n\n".join(logs))


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
