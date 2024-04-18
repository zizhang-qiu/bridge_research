import argparse
import pickle
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iterations", type=int, default=1000000)
    parser.add_argument("--save_dir", type=str, default="score_predictor")
    return parser.parse_args()


def make_sample(cards: np.ndarray, ddts: np.ndarray):
    assert_eq(cards.shape[0], ddts.shape[0])
    data_size = cards.shape[0]
    while True:
        for i in range(data_size):
            trajectory = cards[i]
            deal = rl_cpp.BridgeDeal()
            deal.cards = trajectory
            deal.ddt = ddts[i]
            state = rl_cpp.BridgeBiddingState(deal)
            contract = np.random.randint(1, 420)
            score = state.score_for_contracts(0, [contract])[0]
            scores = [score, -score, score, -score]
            final_obs = state.final_observation_tensor(contract)

            yield torch.tensor(final_obs), torch.tensor(scores)


def make_batch(sample_generator: Generator, batch_size: int, device: str = "cuda"):
    obs_tensor = torch.zeros(size=[batch_size, 229], dtype=torch.float)
    scores_tensor = torch.zeros(size=[batch_size, NUM_PLAYERS], dtype=torch.float)
    while True:
        for batch_idx in range(batch_size):
            obs_tensor[batch_idx], scores_tensor[batch_idx] = next(sample_generator)
        yield obs_tensor.to(device), scores_tensor.to(device)


def main():
    args = parse_args()
    train_dataset = load_rl_dataset("train3")
    valid_dataset = load_rl_dataset("valid")

    train_generator = make_batch(make_sample(train_dataset["cards"], train_dataset["ddts"]), args.batch_size,
                                 args.device)
    valid_generator = make_batch(make_sample(valid_dataset["cards"], valid_dataset["ddts"]), 50000,
                                 args.device)
    net = ScorePredictor()
    net.to(device=args.device)
    opt = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    saver = common_utils.TopKSaver(save_dir, 10)
    stats = common_utils.MultiStats()

    for i in trange(args.num_iterations):
        opt.zero_grad()
        obs, scores = next(train_generator)
        pred = net(obs)
        loss = torch.nn.functional.mse_loss(pred, scores)
        loss.backward()
        opt.step()

        if (i + 1) % args.eval_freq == 0:
            obs, scores = next(valid_generator)
            with torch.no_grad():
                pred = net(obs)
                loss = torch.nn.functional.mse_loss(pred, scores)
                stats.feed("valid loss", loss.item())
                print(loss)
            saver.save(net.state_dict(), -loss)
            stats.save_all(save_dir, True)


def get_all_final_score(dataset: List[List[int]], ddts: np.ndarray):
    for trajectory, ddt in zip(dataset, ddts):
        deal = rl_cpp.BridgeDeal()
        deal.cards = trajectory[:NUM_CARDS]
        deal.ddt = ddt
        state = rl_cpp.BridgeBiddingState(deal)
        for action in trajectory[NUM_CARDS:]:
            state.apply_action(action)


def test_final_score():
    model_dir = "score_predictor/folder_1"
    model_paths = common_utils.find_files_in_dir(model_dir, "model")
    test_dataset = load_rl_dataset("vs_wb5_open_spiel")

    test_generator = make_batch(make_sample(test_dataset["cards"], test_dataset["ddts"]), 10000,
                                "cuda")
    net = ScorePredictor()
    net.to("cuda")
    obs, scores = next(test_generator)
    for model_path in model_paths:
        print(model_path)
        net.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            pred = net(obs).squeeze()
            loss = torch.nn.functional.mse_loss(pred, scores)
            print(loss)
            # print(pred[:50], scores[:50], sep="\n")


if __name__ == '__main__':
    # main()
    test_final_score()
