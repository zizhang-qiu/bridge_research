#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:generate_dataset.py
@time:2023/02/16
"""
import argparse
import os.path
import pickle
import time

import numpy as np

from dds import calc_all_tables, get_ddts_from_dd_table_res_list
from bridge_consts import NUM_CARDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_deals", type=int, default=200000)
    parser.add_argument("--save_dir", type=str, default="dataset/rl_data")
    parser.add_argument("--save_name", type=str, default="value")
    parser.add_argument("--show_progress_bar", action="store_true")

    args = parser.parse_args()
    return args


def main():
    st = time.perf_counter()
    args = parse_args()
    save_name: str = args.save_name
    trajectories = np.zeros([args.num_deals, NUM_CARDS], dtype=int)

    for i in range(args.num_deals):
        trajectory = np.random.permutation(NUM_CARDS)
        trajectories[i] = trajectory
    # print(trajectories)

    double_dummy_results, _ = calc_all_tables(trajectories, show_progress_bar=args.show_progress_bar)
    # print(ddts)
    ddts = get_ddts_from_dd_table_res_list(double_dummy_results)
    # print(ddts)
    print(ddts.shape)
    dataset = {
        "cards": trajectories,
        "ddts": ddts,
        "par_scores": np.zeros(args.num_deals)
    }

    with open(os.path.join(args.save_dir, save_name + ".pkl"), "wb") as fp:
        pickle.dump(dataset, fp)
    ed = time.perf_counter()
    print(f"Generation done! The whole process takes {ed - st:.2f} seconds.")


if __name__ == '__main__':
    main()
