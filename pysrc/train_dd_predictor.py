import argparse
import copy
import os.path
import pickle
from typing import List, Generator, Tuple

import numpy as np
import torch
from adan import Adan
from tqdm import tqdm, trange

import rl_cpp
import common_utils
from bridge_consts import NUM_CARDS
from nets import PolicyNet2, DoubleDummyPredictor
from bridge_bidding_imitation_learning import cross_entropy, compute_accuracy
from utils import load_rl_dataset

OBSERVATION_TENSOR_SIZE = 208


def make_sample(deals: np.ndarray, ddts: np.ndarray) -> Generator:
    """
    Generator for making a sample of (observation tensor, label)
    Args:


    Examples:
        sample_gen = make_sample(train_dataset)
        obs, label = next(sample_gen)

    Returns:
        The generator
    """
    num_deals = deals.shape[0]
    while True:
        for i in range(num_deals):
            cards = deals[i]
            ddt = ddts[i]
            deal = rl_cpp.BridgeDeal()
            deal.cards = cards
            deal.ddt = ddt
            state = rl_cpp.BridgeBiddingState(deal)
            cards_tensor = torch.tensor(state.cards_tensor())
            ddts_tensor = torch.from_numpy(ddt).float()
            yield cards_tensor, ddts_tensor


def make_batch(sample_generator: Generator, batch_size: int, device: str = "cuda"):
    """
    Generator for making batch of (observation tensor, label)
    Args:
        device: where the tensors to put
        sample_generator: The sample generator by make_sample()
        batch_size: The batch size

    Examples:
        batch_gen = make_batch(make_sample(train_dataset), batch_size)
        obs, label = next(batch_gen)

    Returns:
        The generator
    """
    cards_tensor = torch.zeros(size=[batch_size, OBSERVATION_TENSOR_SIZE], dtype=torch.float)
    ddts = torch.zeros(size=(batch_size, 20), dtype=torch.float)
    while True:
        for batch_idx in range(batch_size):
            cards_tensor[batch_idx], ddts[batch_idx] = next(sample_generator)
        yield cards_tensor.to(device), ddts.to(device)


def make_all_samples(deals: np.ndarray, ddts: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    cards_tensors = []
    ddts_tensors = []
    num_deals = deals.shape[0]
    for i in trange(num_deals):
        deal = rl_cpp.BridgeDeal()
        deal.cards = deals[i]
        deal.ddt = ddts[i]
        state = rl_cpp.BridgeBiddingState(deal)
        cards_tensor = torch.tensor(state.cards_tensor())
        cards_tensors.append(cards_tensor)
        ddt_tensor = torch.from_numpy(ddts[i]).float()
        ddts_tensors.append(ddt_tensor)
    return torch.vstack(cards_tensors).to(device), torch.vstack(ddts_tensors).to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--num_iterations", type=int, default=500000)
    return parser.parse_args()


def main():
    args = parse_args()
    common_utils.set_random_seeds(42)
    device = "cuda"
    net = DoubleDummyPredictor()
    common_utils.initialize_fc(net)
    net.to(device)
    opt = torch.optim.Adam(params=net.parameters(), lr=args.lr, fused=True)
    save_dir = common_utils.mkdir_with_increment("double_dummy_predictor")
    saver = common_utils.TopKSaver(save_dir, 10)

    train_dataset = load_rl_dataset("train")
    train_generator = make_batch(make_sample(train_dataset["cards"], train_dataset["ddts"]), args.batch_size, device)

    valid_dataset = load_rl_dataset("valid")
    valid_cards_tensor, valid_ddts_tensor = make_all_samples(valid_dataset["cards"], valid_dataset["ddts"], device)
    print(valid_cards_tensor.shape, valid_ddts_tensor.shape)

    for i_iter in trange(args.num_iterations):
        opt.zero_grad()
        train_cards_tensor, train_ddts_tensor = next(train_generator)
        pred = net(train_cards_tensor)
        loss = torch.nn.functional.mse_loss(pred, train_ddts_tensor)
        loss.backward()
        opt.step()
        if (i_iter + 1) % args.eval_freq == 0:
            with torch.no_grad():
                pred = net(valid_cards_tensor)
                valid_loss = torch.nn.functional.mse_loss(pred, valid_ddts_tensor)
                print(f"iteration {i_iter + 1}, valid loss={valid_loss}.")
                saver.save(copy.deepcopy(net.state_dict()), -valid_loss.item())


def test():
    models_dir = "double_dummy_predictor/folder_1"
    models_paths = common_utils.find_files_in_dir(models_dir, "model4")
    net = DoubleDummyPredictor()
    net.to("cuda")
    valid_dataset = load_rl_dataset("valid")
    valid_cards_tensor, valid_ddts_tensor = make_all_samples(valid_dataset["cards"], valid_dataset["ddts"], "cuda")
    for model_path in models_paths:
        net.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            pred = net(valid_cards_tensor)
            valid_loss = torch.nn.functional.mse_loss(pred, valid_ddts_tensor)
            rounded_pred = torch.round(pred)
            rounded_loss = torch.nn.functional.mse_loss(rounded_pred, valid_ddts_tensor)
            print(f"{model_path}, valid loss={valid_loss}.")
            print(rounded_loss)
            print(pred, valid_ddts_tensor)
            print(rounded_pred)


if __name__ == '__main__':
    # main()
    test()
