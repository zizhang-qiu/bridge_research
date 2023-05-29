import argparse
import copy
import os.path
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
from adan import Adan
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from nets import PerfectValueNet
import rl_cpp
from pysrc import common_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_batch_size", type=int, default=128)
    # parser.add_argument("--valid_batch_size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="value_sl")
    parser.add_argument("--eval_freq", type=int, default=1000)
    return parser.parse_args()


class ValueDataset(Dataset):
    def __init__(self, dataset_path: str, device: str):
        if not os.path.exists(dataset_path):
            raise ValueError(f"The path {dataset_path} doesn't exist.")

        dataset: Dict[str, torch.Tensor] = torch.load(dataset_path)
        self.perfect_obs = dataset["perfect_obs"]
        self.labels = dataset["labels"] / 7600
        self.device = device

    def __len__(self):
        return self.perfect_obs.shape[0]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.perfect_obs[item].to(self.device), self.labels[item].to(self.device)


def make_dataset(trajectories: List[List[int]], scores: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    perfect_obs_list = []
    labels_list = []

    num_deals = len(trajectories)
    assert (num_deals == scores.size)
    for i in trange(num_deals):
        trajectory = trajectories[i]
        deal = rl_cpp.BridgeDeal()
        deal.cards = trajectory[:52]
        deal.ddt = np.zeros(20, dtype=int)
        state = rl_cpp.BridgeBiddingState(deal)
        num_actions = len(trajectory) - 52
        perfect_observations = torch.zeros([num_actions, 636], dtype=torch.float)
        labels = torch.zeros(num_actions, dtype=torch.float)
        for j, action in enumerate(trajectory[52:]):
            hidden_observation = state.hidden_observation_tensor()
            observation = state.observation_tensor()
            observation.extend(hidden_observation)
            # print(observation)
            state.apply_action(action - 52)
            perfect_observations[j] = torch.tensor(observation)
        # print(perfect_observations)
        score = scores[i]

        for k in range(num_actions):
            labels[k] = ((-1) ** k) * score
        # print(labels)
        # print(state)
        perfect_obs_list.append(perfect_observations)
        labels_list.append(labels)
    perfect_obs_tensor = torch.vstack(perfect_obs_list)
    labels_obs_tensor = torch.hstack(labels_list)
    print(perfect_obs_tensor, labels_obs_tensor)
    print(perfect_obs_tensor.shape, labels_obs_tensor.shape)
    return perfect_obs_tensor, labels_obs_tensor


def main():
    args = parse_args()
    # category = "train"
    # with open(f"dataset/expert/{category}.pkl", "rb") as fp:
    #     data = pickle.load(fp)
    # # print(data)
    # scores = np.load(fr"D:\RL\rlul\pyrlul\bridge\dataset\expert\{category}_scores.npy")
    # perfect_obs, labels = make_dataset(data, scores)
    # torch.save(
    #     {
    #         "perfect_obs": perfect_obs,
    #         "labels": labels
    #     }, os.path.join("dataset/expert", f"perfect_{category}.p")
    # )
    common_utils.set_random_seeds(1)
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    logger = common_utils.Logger(os.path.join(save_dir, "log.txt"), auto_line_feed=True)
    saver = common_utils.TopKSaver(5, save_dir, "value")
    net = PerfectValueNet()
    net.to(args.device)
    valid_dataset = ValueDataset("dataset/expert/perfect_valid.p", args.device)
    train_dataset = ValueDataset("dataset/expert/perfect_train.p", args.device)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
    num_mini_batches = 0
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=True)

    def evaluate():
        with torch.no_grad():
            for valid_perfect_obs, valid_labels in valid_dataloader:
                pred = net(valid_perfect_obs).squeeze()
                valid_loss = torch.nn.functional.mse_loss(pred, valid_labels)
        msg = f"checkpoint {num_mini_batches // args.eval_freq}, valid loss: {valid_loss}."
        saver.save(copy.deepcopy(net.state_dict()), -valid_loss.item(), True)
        logger.write(msg)

    def train():
        nonlocal num_mini_batches
        for perfect_obs, labels in train_dataloader:
            # print(labels)
            print(f"\rnum_mini_batches: {num_mini_batches}", end="")
            opt.zero_grad()
            pred = net(perfect_obs).squeeze()
            loss = torch.nn.functional.mse_loss(pred, labels)
            loss.backward()
            opt.step()
            num_mini_batches += 1
            if num_mini_batches % args.eval_freq == 0:
                evaluate()

    while True:
        train()


def test():
    torch.set_printoptions(threshold=1000000, sci_mode=False)
    net = PerfectValueNet()
    net.load_state_dict(torch.load("../value_sl/folder_3/value_0.pth"))
    net.to("cuda")
    test_dataset = ValueDataset("../expert/perfect_test.p", "cuda")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    with torch.no_grad():
        for perfect_obs, labels in test_loader:
            pred = net(perfect_obs).squeeze()
            test_loss = torch.nn.functional.mse_loss(pred, labels, reduction="mean")
            print(test_loss)
            print(pred * 7600, labels * 7600)
            # for p, l in zip(pred, labels):
            #     print(p, l)


if __name__ == '__main__':
    test()
    # main()
