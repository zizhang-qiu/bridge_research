import argparse
import copy
import os
import pprint
from typing import Tuple

import numpy as np
import torch
import torchmetrics
from adan import Adan
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from nets import PolicyNet
from common_utils.logger import Logger
from common_utils.other_utils import set_random_seeds, mkdir_with_time
from common_utils.torch_utils import initialize_fc
from common_utils.value_stats import MultiStats
from bridge_vars import NUM_CALLS
import common_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
NUM_ACTIONS = 38


class BiddingDataset(Dataset):
    def __init__(self, obs_path: str, label_path: str):
        """
        The dataset contains bridge bidding data.
        Args:
            obs_path: The path of obs.
            label_path: The path of labels.
        """
        self.s: torch.Tensor = torch.load(obs_path)["s"]
        self.label: torch.Tensor = torch.load(label_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.s[index], self.label[index]


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset path
    parser.add_argument("--dataset_dir", type=str,
                        default=r"D:\RL\rlul\pyrlul\bridge\dataset\expert\sl_data")

    # save settings
    parser.add_argument("--save_dir", type=str, default="imitation_learning")

    # train settings
    parser.add_argument("--trained_checkpoint", type=str,
                        default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_mini_batches", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=40)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def cross_entropy(log_probs: torch.Tensor, label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute cross entropy loss of given log probs and label.
    Args:
        log_probs: The log probs.
        label: The label, should be 1 dimensional.
        num_classes: The number of classes for one-hot.

    Returns:
        The cross entropy loss.
    """
    assert label.ndimension() == 1
    return -torch.mean(torch.nn.functional.one_hot(label.long(), num_classes) * log_probs)


def compute_accuracy(probs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy of given probs and label. Which is the number of highest value action equals with label
    divides number of all actions.
    Args:
        probs: The probs.
        label: The labels.

    Returns:
        The accuracy of prediction.
    """
    greedy_actions = torch.argmax(probs, 1)
    return (greedy_actions == label).int().sum() / greedy_actions.shape[0]


def main():
    args = parse_args()
    set_random_seeds(args.seed)
    train_dataset = BiddingDataset(obs_path=os.path.join(args.dataset_dir, "train_obs.p"),
                                   label_path=os.path.join(args.dataset_dir, "train_label.p"))
    valid_dataset = BiddingDataset(obs_path=os.path.join(args.dataset_dir, "valid_obs.p"),
                                   label_path=os.path.join(args.dataset_dir, "valid_label.p"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)
    print("load dataset successfully!")

    # create save_dir
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    saver = common_utils.TopKSaver(10, save_dir, "checkpoint")

    # initialize network, optimizer and criterion
    net = PolicyNet()
    initialize_fc(net)
    net.to(args.device)
    # opt = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    opt = Adan(params=net.parameters(), lr=args.lr, fused=True)
    if args.trained_checkpoint:
        checkpoint = torch.load(args.trained_checkpoint)
        net.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
    # loss_func = torch.nn.BCELoss()
    # value stats and logger
    multi_stats = MultiStats()

    logger = Logger(os.path.join(save_dir, "log.txt"), verbose=True, auto_line_feed=True)
    logger.write(pprint.pformat(args))
    num_mini_batches = 0

    #

    def train_epoch():
        nonlocal num_mini_batches
        while True:
            # loss_list = []
            for s, label in train_loader:
                # print(s, label)
                num_mini_batches += 1
                opt.zero_grad()
                s = s.to(args.device)
                label = label.to(args.device)
                log_prob = net(s)
                loss = cross_entropy(log_prob, label, NUM_ACTIONS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.grad_clip)
                opt.step()
                # loss_list.append(loss.item())
                # eval
                if num_mini_batches % args.eval_freq == 0:
                    # multi_stats.save("train_loss", save_dir)
                    eval_loss, acc = evaluate()
                    multi_stats.feed("eval_loss", eval_loss)
                    multi_stats.feed("accuracy", acc)
                    multi_stats.save_all(save_dir, True)
                    msg = f"checkpoint {(num_mini_batches + 1) // args.eval_freq}, eval loss={eval_loss}, accuracy={acc}"
                    logger.write(msg)
                    # save params
                    check_point = {'model_state_dict': copy.deepcopy(net.state_dict())}
                    saver.save(check_point, acc)
                    # torch.save(check_point,
                    #            os.path.join(save_dir, f"checkpoint{num_mini_batches // args.eval_freq}.pth"))
                if num_mini_batches == args.max_mini_batches:
                    return

    def evaluate() -> Tuple[float, float]:
        loss_list = []
        acc_list = []
        with torch.no_grad():
            for s, label in valid_loader:
                s = s.to(args.device)
                label = label.to(args.device)
                log_prob = net(s)
                loss = cross_entropy(log_prob, label, NUM_ACTIONS)
                loss_list.append(loss.item())
                accuracy = compute_accuracy(torch.exp(log_prob), label)
                acc_list.append(accuracy.item())
        return np.mean(loss_list).item(), np.mean(acc_list).item()

    train_epoch()
    # multi_stats.feed("train_loss", epoch_loss)


def test():
    args = parse_args()
    test_dataset = BiddingDataset(obs_path=os.path.join(args.dataset_dir, "test_obs.p"),
                                  label_path=os.path.join(args.dataset_dir, "test_label.p"))
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    net = PolicyNet()
    net.load_state_dict(torch.load(args.trained_checkpoint)["model_state_dict"])
    net.to(args.device)

    def _test() -> Tuple[float, float]:
        loss_list = []
        acc_list = []
        with torch.no_grad():
            for s, label in test_loader:
                s = s.to(args.device)
                label = label.to(args.device)
                log_prob = net(s)
                loss = cross_entropy(log_prob, label, NUM_ACTIONS)
                loss_list.append(loss.item())
                accuracy = compute_accuracy(torch.exp(log_prob), label)
                acc_list.append(accuracy.item())
        return np.mean(loss_list).item(), np.mean(acc_list).item()

    test_loss, test_acc = _test()
    print(f"test loss:{test_loss}, test accuracy:{test_acc}")


if __name__ == '__main__':
    # test_and_compute_metrics()
    main()
