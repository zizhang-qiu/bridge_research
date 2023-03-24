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
import torch.multiprocessing as mp

import torch

import rl_cpp

from src.bridge.bluechip_bridge import Controller, BlueChipBridgeBot
from src.bridge.agent_for_cpp import SingleEnvAgent
from src.bridge.bridge_vars import NUM_PLAYERS
from src.bridge.nets import PolicyNet
from src.bridge.utils import load_rl_dataset
from src.common_utils.assert_utils import assert_eq
from src.common_utils.logger import Logger
from src.common_utils.other_utils import mkdir_with_time
from src.common_utils.torch_utils import create_shared_dict


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
                 shared_dict: Dict, device: str, save_dir: Optional[str] = None, imps_queue: Optional[mp.Queue] = None):
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
            imps_queue: The queue to store imps.
        """
        super().__init__()
        assert_eq(num_deals, cards.shape[0])
        assert_eq(num_deals, ddts.shape[0])
        self._process_id = process_id
        self._num_deals = num_deals
        self._cards = cards
        self._ddts = ddts
        self._port = port
        self._device = device
        self.shared_dict = shared_dict.copy()
        self.logger: Optional[Logger] = None
        self.save_dir = save_dir if save_dir is not None else None
        self.imps_queue = imps_queue

    def run(self) -> None:
        p_net = PolicyNet()
        p_net.load_state_dict(self.shared_dict)
        agent = SingleEnvAgent(p_net)
        agent.to(self._device)
        if self.save_dir is not None:
            f = open(os.path.join(self.save_dir, f"log_{self._process_id}.txt"), "w")
        i_deal = 0
        _imps = []
        bots = [BlueChipBridgeBot(i, controller_factory, self._port + i) for i in range(NUM_PLAYERS)]
        while i_deal < self._num_deals:
            try:
                state_0 = rl_cpp.BridgeBiddingState(0, self._cards[i_deal], False, False, self._ddts[i_deal])
                state_1 = rl_cpp.BridgeBiddingState(0, self._cards[i_deal], False, False, self._ddts[i_deal])
                for bot in bots:
                    bot.restart()

                while not state_0.terminated():
                    current_player = state_0.current_player()
                    if current_player % 2 == 0:
                        obs = rl_cpp.make_obs_tensor(state_0, 1)
                        obs = obs.to(self._device)
                        action, _ = agent.act(obs)
                        action = action.item()
                    else:
                        action = bots[current_player].step(state_0)
                    # print(action)
                    state_0.apply_action(action)

                # print(env_0)
                while not state_1.terminated():
                    current_player = state_1.current_player()
                    if current_player % 2 == 1:
                        obs = rl_cpp.make_obs_tensor(state_1, 1)
                        obs = obs.to(self._device)
                        action, _ = agent.act(obs)
                        action = action.item()
                    else:
                        action = bots[current_player].step(state_1)
                    state_1.apply_action(action)

                imp = rl_cpp.get_imp(int(state_0.returns()[0]), int(state_1.returns()[0]))
                _imps.append(imp)
                print(imp)
                if self.imps_queue is not None:
                    self.imps_queue.put(imp)
                    print(f"qsize:{self.imps_queue.qsize()}")
                msg = f"open:\n{state_0}\n\nclose:\n{state_1}\n\nimp: {imp}\n\n"
                if self.save_dir is not None:
                    f.write(msg)
                # print(msg)
                i_deal += 1
            except Exception:
                continue

        if self.save_dir is not None:
            _imps_np = np.array(_imps)
            np.save(os.path.join(self.save_dir, f"imps_{self._process_id}.npy"))


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    cards, ddts = load_rl_dataset("vs_wb5", flatten=True)
    net = PolicyNet()
    net.load_state_dict(torch.load(r"/src/policy_gradient/20230316170427/checkpoint_1737.pth")
                        ["model_state_dict"]["policy"])
    net.to("cuda")
    shared_dict = create_shared_dict(net)
    num_deals = 100
    save_dir = mkdir_with_time("../vs_wbridge5")
    q = mp.Queue()
    base_port = 1
    workers = []
    for i in range(8):
        w = AgainstWb5Worker(i, num_deals, base_port + 10 * i, cards[num_deals * i:num_deals * (i + 1)],
                             ddts[num_deals * i:num_deals * (i + 1)], shared_dict, "cuda", save_dir, q)
        workers.append(w)
    for w in workers:
        w.start()

    for w in workers:
        w.join()
