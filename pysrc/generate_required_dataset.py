import pickle

import numpy as np

import dds
import common_utils
import torch
import rl_cpp

from bridge_consts import Denomination, NUM_DENOMINATIONS, NUM_CARDS


def get_bid(level: int, trump: Denomination) -> int:
    return (level - 1) * NUM_DENOMINATIONS + trump


def main():
    common_utils.set_random_seeds(1)
    required_contracts = list()
    required_contracts.append(get_bid(3, Denomination.NO_TRUMP))
    required_contracts.append(get_bid(4, Denomination.NO_TRUMP))
    required_contracts.append(get_bid(5, Denomination.NO_TRUMP))
    required_contracts.append(get_bid(6, Denomination.NO_TRUMP))
    required_contracts.append(get_bid(4, Denomination.CLUBS))
    required_contracts.append(get_bid(4, Denomination.DIAMONDS))
    required_contracts.append(get_bid(4, Denomination.HEARTS))
    required_contracts.append(get_bid(4, Denomination.SPADES))
    required_contracts.append(get_bid(5, Denomination.CLUBS))
    required_contracts.append(get_bid(5, Denomination.DIAMONDS))
    required_contracts.append(get_bid(5, Denomination.HEARTS))
    required_contracts.append(get_bid(5, Denomination.SPADES))
    required_contracts.append(get_bid(6, Denomination.CLUBS))
    required_contracts.append(get_bid(6, Denomination.DIAMONDS))
    required_contracts.append(get_bid(6, Denomination.HEARTS))
    required_contracts.append(get_bid(6, Denomination.SPADES))

    cards = np.zeros([1000, NUM_CARDS], dtype=int)
    for i in range(1000):
        cards[i] = np.random.permutation(NUM_CARDS)

    ddt, par_scores = dds.calc_all_tables(cards)
    par_scores_list, par_contracts_list = dds.get_par_scores_and_contracts_from_pres_list(par_scores)
    print(par_scores_list, par_contracts_list)


if __name__ == '__main__':
    main()
