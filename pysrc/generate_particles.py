import argparse
import os.path
import pickle
from typing import Union, List

import numpy as np
import torch
from tqdm import trange

import rl_cpp
import utils
import dds
import common_utils
from bridge_consts import NUM_PLAYERS, NUM_CARDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vs_wb5_open_spiel")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="dataset/fine_tune")
    parser.add_argument("--save_name", type=str, default="1")
    return parser.parse_args()


def state_from_cards(cards: Union[np.ndarray, List[int]]):
    deal = rl_cpp.BridgeDeal()
    deal.cards = cards
    state = rl_cpp.BridgeBiddingState(deal)
    return state


def sample_one_deal(cards: np.ndarray, num_samples: int, rng: rl_cpp.RNG):
    state = state_from_cards(cards)
    result_cards = np.ndarray([NUM_PLAYERS, num_samples, NUM_CARDS], dtype=int)
    result_ddts = np.ndarray([NUM_PLAYERS, num_samples, 20], dtype=int)
    for player in range(NUM_PLAYERS):
        player_cards = state.get_player_cards(player)
        for i in range(num_samples):
            particle = rl_cpp.jointly_sample(player_cards, player, rng)
            result_cards[player][i] = particle
        dd_table_res_list, _ = dds.calc_all_tables(result_cards[player], False)
        result_ddts[player] = dds.get_ddts_from_dd_table_res_list(dd_table_res_list)
    return result_cards, result_ddts


def main():
    args = parse_args()
    dataset = utils.load_rl_dataset(args.dataset)
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    # print(dataset)
    cards = dataset["cards"]
    rng = rl_cpp.RNG(1)
    num_deals = cards.shape[0]
    result_cards = np.zeros([num_deals, NUM_PLAYERS, args.num_samples, NUM_CARDS], dtype=int)
    result_ddts = np.zeros([num_deals, NUM_PLAYERS, args.num_samples, 20], dtype=int)
    for i in trange(num_deals):
        # sample_cards shape[4, num_samples, 52], sample_ddts shape [4, num_samples, 20]
        sample_cards, sample_ddts = sample_one_deal(cards[i], args.num_samples, rng)
        result_cards[i] = sample_cards
        result_ddts[i] = sample_ddts
    save_dataset = {
        "cards": result_cards,
        "ddts": result_ddts
    }
    with open(os.path.join(save_dir, args.save_name) + ".pkl", "wb") as fp:
        pickle.dump(save_dataset, fp)


if __name__ == '__main__':
    main()
