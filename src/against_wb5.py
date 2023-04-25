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
from typing import Dict, Optional

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
from utils import load_rl_dataset, tensor_dict_to_device


class _WBridge5Client(Controller):
    """Manages the connection to a WBridge5 bot."""

    def __init__(self, command, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("localhost", port))
        self.port = port
        self.sock.listen(5)
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
            self.conn.settimeout(60)
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
        self.shared_dict = shared_dict.copy()
        self.logger: Optional[common_utils.Logger] = None
        self.save_dir = save_dir

    def run(self) -> None:
        p_net = PolicyNet()
        p_net.load_state_dict(self.shared_dict)
        agent = SingleEnvAgent(p_net)
        agent.to(self._device)
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
                        action = reply["a"].item()
                    else:
                        action = bots[current_player].step(state_0)
                    # print(action)
                    state_0.apply_action(action)

                # print(env_0)
                while not state_1.terminated():
                    current_player = state_1.current_player()
                    if current_player % 2 == 1:
                        obs = rl_cpp.make_obs_tensor_dict(state_1, 1)
                        obs = tensor_dict_to_device(obs, self._device)
                        reply = agent.act(obs)
                        action = reply["a"].item()
                    else:
                        action = bots[current_player].step(state_1)
                    state_1.apply_action(action)

                imp = rl_cpp.get_imp(int(state_0.returns()[0]), int(state_1.returns()[0]))
                _imps.append(imp)
                msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\nimp: {imp}\n\n"
                with open(os.path.join(self.save_dir, f"log_{self._process_id}.txt"), "w") as f:
                    f.write(msg)
                _imps_np = np.array(_imps)
                np.save(os.path.join(self.save_dir, f"imps_{self._process_id}.npy"), _imps_np)
                # print(msg)
                i_deal += 1
                print(f"Process {self._process_id}, {i_deal}/{self._num_deals}, imp:{imp}")
            except Exception as e:
                print(f"Process {self._process_id} meet exception: {e}.")
                continue


def main():
    # cards, ddts = load_rl_dataset("vs_wb5_fb", flatten=True)
    dataset = load_rl_dataset("vs_wb5_open_spiel")
    cards = dataset["cards"]
    ddts = dataset["ddts"]
    with open("config/against_wb5.yaml", "r") as fp:
        config: Dict = yaml.safe_load(fp)
    # print(config)
    net = PolicyNet()
    net.load_state_dict(torch.load(config["checkpoint_path"])["model_state_dict"]["policy"])
    net.to("cuda")
    num_processes = config["num_processes"]
    num_deals_per_process = common_utils.allocate_tasks_uniformly(num_processes, config["num_deals"])
    shared_dict = common_utils.create_shared_dict(net)
    save_dir = common_utils.mkdir_with_increment(config["save_dir"])
    base_port = 5050
    start = config["start"]
    # w = AgainstWb5Worker(0, 100, base_port, cards[:100], ddts[:100], shared_dict, "cuda", save_dir)
    # w.run()

    workers = []
    for i in range(config["num_processes"]):
        num_deals = num_deals_per_process[i]
        w = AgainstWb5Worker(i, num_deals,
                             base_port + 10 * i,
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
    print(f"result is {avg}{PLUS_MINUS_SYMBOL}{sem}")
    np.save(os.path.join(save_dir, "imps.npy"), imps)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
