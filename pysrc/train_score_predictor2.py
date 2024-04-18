import argparse
import pickle
import time
from typing import List, Generator

import numpy as np
import torch
from tqdm import trange

import rl_cpp

from nets import ScorePredictor
from bridge_consts import NUM_CARDS, NUM_PLAYERS
import common_utils
from common_utils import assert_eq
from utils import load_rl_dataset


def main():
    dataset = load_rl_dataset("train3")
    deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"])
    replay = rl_cpp.FinalObsScoreReplay(1000000, 1, 0.0, 0.0, 1)
    ctx = rl_cpp.Context()
    for i in range(4):
        t = rl_cpp.ContractScoreThreadLoop(deal_manager, replay, 10, i)
        ctx.push_thread_loop(t)
    ctx.start()
    time.sleep(5)
    final_obs_score = replay.sample(10, "cuda")
    print(final_obs_score.final_obs)
    print(final_obs_score.score)


if __name__ == '__main__':
    main()
