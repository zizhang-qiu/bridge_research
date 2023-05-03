import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

import rl_cpp
from bridge_vars import NUM_CARDS

parser = argparse.ArgumentParser()
# dataset path
parser.add_argument("--dataset_dir", type=str,
                    default=r"D:\RL\rlul\pyrlul\bridge\dataset\expert\sl_data")
parser.add_argument("--name", type=str, default="train")
parser.add_argument("--save_dir", type=str, default="dataset/expert")
args = parser.parse_args()


def main():
    obs = torch.load(os.path.join(args.dataset_dir, f"{args.name}_obs.p"))
    labels = torch.load(os.path.join(args.dataset_dir, f"{args.name}_label.p"))
    print(obs)
    print(labels)


if __name__ == '__main__':
    main()
