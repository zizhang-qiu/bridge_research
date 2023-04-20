#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:generate_dataset.py
@time:2023/02/16
"""
import argparse
import os.path
import time

import numpy as np

from dds import calc_all_tables
from bridge_vars import NUM_CARDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_deals", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="../dataset/rl_data")
    parser.add_argument("--save_name", type=str, default="vs_wb5")
    parser.add_argument("--show_progress_bar", action="store_true")

    args = parser.parse_args()
    return args


def main():
    st = time.perf_counter()
    args = parse_args()
    save_name: str = args.save_name
    suffix = "" if save_name.endswith(".npy") else ".npy"
    trajectories_save_path = os.path.join(args.save_dir, save_name + "_trajectories" + suffix)
    ddts_save_path = os.path.join(args.save_dir, save_name + "_ddts" + suffix)

    trajectories = np.zeros([args.num_deals, NUM_CARDS], dtype=int)

    for i in range(args.num_deals):
        trajectory = np.random.permutation(NUM_CARDS)
        trajectories[i] = trajectory

    ddts = calc_all_tables(trajectories, show_progress_bar=args.show_progress_bar)
    # print(ddts)
    np.save(trajectories_save_path, trajectories)
    np.save(ddts_save_path, ddts)
    ed = time.perf_counter()
    print(f"Generation done! The whole process takes {ed - st:.2f} seconds.")


if __name__ == '__main__':
    main()
