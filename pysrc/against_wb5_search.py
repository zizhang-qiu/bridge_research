"""
@file:against_wb5
@author:qzz
@date:2023/3/4
@encoding:utf-8
"""
import os
import re
import socket
import subprocess
import time
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

import common_utils
import rl_cpp
from agent_for_cpp import SingleEnvAgent
from bluechip_bridge import Controller, BlueChipBridgeBot
from bridge_vars import NUM_PLAYERS, PLUS_MINUS_SYMBOL
from nets import PolicyNet
from utils import load_rl_dataset, tensor_dict_to_device, sl_net


class _WBridge5Client(Controller):
    """Manages the connection to a WBridge5 bot."""

    def __init__(self, command, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("localhost", port))
        self.port = port
        self.sock.listen(1)
        self.process = None
        self.command = command.format(port=self.port)

    def start(self):
        """
        Open the external bot and connect with it
        Returns:

        """
        if self.process is not None:
            self.process.kill()
        self.process = subprocess.Popen(self.command.split(" "))
        self.conn, self.addr = self.sock.accept()

    def read_line(self) -> str:
        """
        Read message from wb5
        Returns:
        str: the message
        """
        line = ""
        while True:
            self.conn.settimeout(120)
            data = self.conn.recv(2048)
            if not data:
                raise EOFError("Connection closed")
            line += data.decode("ascii")
            if line.endswith("\n"):
                return re.sub(r"\s+", " ", line).strip()

    def send_line(self, line: str):
        """
        Send message to wbridge5
        Args:
            line(str):the message to send

        Returns:

        """
        self.conn.send((line + "\r\n").encode("ascii"))

    def terminate(self):
        """
        Terminate process
        Returns:

        """
        self.process.kill()
        self.process = None
        self.sock.close()


def controller_factory(port: int):
    """Implements bluechip_bridge.BlueChipBridgeBot."""
    client = _WBridge5Client("d:/wbridge5/Wbridge5.exe Autoconnect {port}", port=port)
    client.start()
    return client


class AgainstWb5Worker(mp.Process):
    def __init__(self, process_id: int, num_deals: int, port: int, cards: np.ndarray, ddts: np.ndarray,
                 shared_dicts: Dict[str, Dict], search_config: Dict, device: str, save_dir: Optional[str] = None):
        """
        A worker to play against Wbridge5.
        Args:
            process_id: The id of the process
            num_deals: The number of deals to play
            port: The port number to connect with Wbridge5
            cards: The cards array.
            ddts: The ddts array.
            shared_dicts: The shared dict of policy network.
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
        self._search_config = search_config
        self._shared_dicts = {k: v.copy() for k, v in shared_dicts.items()}
        self.logger: Optional[common_utils.Logger] = None
        self.save_dir = save_dir

    def run(self) -> None:
        actors_ns = []
        actors_ew = []
        trained_net = PolicyNet()
        trained_net.load_state_dict(self._shared_dicts["trained"])
        trained_model_locker = rl_cpp.ModelLocker(
            [torch.jit.script(SingleEnvAgent(trained_net)).to(self._device)], self._device)
        if self._search_config["use_sl_net_as_opponent"]:
            supervised_net = PolicyNet()
            supervised_net.load_state_dict(self._shared_dicts["sl"])
            sl_model_locker = rl_cpp.ModelLocker(
                [torch.jit.script(SingleEnvAgent(supervised_net)).to(self._device)], self._device)
            for i in range(2):
                actors_ns.append(rl_cpp.SingleEnvActor(trained_model_locker))
                actors_ns.append(rl_cpp.SingleEnvActor(sl_model_locker))
                actors_ew.append(rl_cpp.SingleEnvActor(sl_model_locker))
                actors_ew.append(rl_cpp.SingleEnvActor(trained_model_locker))

        else:
            actors_ns = [rl_cpp.SingleEnvActor(trained_model_locker) for _ in range(NUM_PLAYERS)]
            actors_ew = [rl_cpp.SingleEnvActor(trained_model_locker) for _ in range(NUM_PLAYERS)]
        net = PolicyNet()
        net.load_state_dict(self._shared_dicts["trained"])
        net.to(self._device)
        agent = SingleEnvAgent(net).to(self._device)
        params = rl_cpp.SearchParams()
        params.min_prob = self._search_config["min_prob"]
        params.max_particles = self._search_config["max_particles"]
        params.max_rollouts = self._search_config["max_rollouts"]
        params.min_rollouts = self._search_config["min_rollouts"]
        params.verbose_level = self._search_config["verbose_level"]
        params.seed = self._search_config["seed"]
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
                        reply = agent.act(obs)
                        # action = reply["a"].item()
                        probs = torch.exp(reply["log_probs"]) * (obs["legal_actions"].cpu())
                        # print("probs: ", probs)
                        action = rl_cpp.search(probs, state_0, actors_ns, params)
                        print(action)
                    else:
                        action = bots[current_player].step(state_0)
                    # print(action)
                    state_0.apply_action(action)

                print(state_0)
                while not state_1.terminated():
                    current_player = state_1.current_player()
                    if current_player % 2 == 1:
                        obs = rl_cpp.make_obs_tensor_dict(state_1, 1)
                        obs = tensor_dict_to_device(obs, self._device)
                        reply = agent.act(obs)
                        # action = reply["a"].item()
                        probs = torch.exp(reply["log_probs"]) * (obs["legal_actions"].cpu())
                        # print("probs: ", probs)
                        action = rl_cpp.search(probs, state_1, actors_ew, params)
                        print(action)
                    else:
                        action = bots[current_player].step(state_1)
                    state_1.apply_action(action)

                print(state_1)

                imp = rl_cpp.get_imp(int(state_0.returns()[0]), int(state_1.returns()[0]))
                _imps.append(imp)
                msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\nimp: {imp}\n\n"
                with open(os.path.join(self.save_dir, f"log_{self._process_id}.txt"), "a") as f:
                    f.write(msg)
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
    with open("conf/against_wb5_search.yaml", "r") as fp:
        config: Dict = yaml.safe_load(fp)
    search_cfg = config["search"]
    net = PolicyNet()
    net.load_state_dict(torch.load(config["checkpoint_path"])["model_state_dict"]["policy"])
    net.to("cuda")
    supervised_net = sl_net()
    num_processes = config["num_processes"]
    num_deals_per_process = common_utils.allocate_tasks_uniformly(num_processes, config["num_deals"])
    shared_dict = common_utils.create_shared_dict(net)
    sl_shared_dict = common_utils.create_shared_dict(supervised_net)
    shared_dicts = {
        "sl": sl_shared_dict,
        "trained": shared_dict
    }
    save_dir = common_utils.mkdir_with_increment(config["save_dir"])
    base_port = 5050
    start = config["start"]
    st = time.perf_counter()
    w = AgainstWb5Worker(0, 100, base_port, cards[:100], ddts[:100], shared_dicts, search_cfg, "cuda", save_dir)
    w.run()
    print(time.perf_counter() - st)

    # workers = []
    # for i in range(num_processes):
    #     num_deals = num_deals_per_process[i]
    #     w = AgainstWb5Worker(i, num_deals,
    #                          base_port + 10 * i,
    #                          cards[start + num_deals * i:start + num_deals * (i + 1)],
    #                          ddts[start + num_deals * i:start + num_deals * (i + 1)],
    #                          shared_dicts, search_cfg, "cuda", save_dir)
    #     workers.append(w)
    # for w in workers:
    #     w.start()
    #
    # for w in workers:
    #     w.join()

    # get average and standard error of mean
    # imps_list = [np.load(os.path.join(save_dir, f"imps_{i}.npy")) for i in range(num_processes)]
    # imps = np.concatenate(imps_list)
    # avg, sem = common_utils.get_avg_and_sem(imps)
    # print(f"result is {avg}{PLUS_MINUS_SYMBOL}{sem}")
    # np.save(os.path.join(save_dir, "imps.npy"), imps)
    # print(f"The whole process takes {time.perf_counter() - st:.2f} seconds.")


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
