"""
@file:make_rl_dataset_from_db
@author:qzz
@date:2023/3/21
@encoding:utf-8
"""
import os
import pickle
import re
import sqlite3

import numpy as np
from tqdm import trange

from src.bridge.dds import calc_all_tables, get_par_score_from_par_results
from src.bridge.bridge_vars import NUM_CARDS, RANK_STR, NUM_CARDS_PER_HAND, NUM_PLAYERS
from src.bridge.pbn import get_card, cards_to_pbn, get_pbn_game_string, create_pbn_file
from src.bridge.utils import load_rl_dataset

db_dir = r"D:\RL\rlul\pyrlul\bridge\dataset\rl_data"
db_name = "train"


def get_card_and_ddt_from_str(db_str: str):
    cards = re.search('\\"N:(.*?)\"', db_str).group(1)[:-1].split(' ')
    ddt_str = (re.search('"ddt": \[(.*?)]', db_str).group(1)).split(',')
    ddt = [int(num) for num in ddt_str]
    # print(cards, ddt)
    cards_actions = []
    for card in cards:
        card_per_suit = card.split('.')
        cards_action = []
        for suit, ranks in enumerate(card_per_suit):
            for rank in ranks:
                cards_action.append(get_card(3 - suit, RANK_STR.index(rank)))
        cards_actions.append(cards_action)

    cards_trajectory = []
    for i in range(NUM_CARDS_PER_HAND):
        for j in range(NUM_PLAYERS):
            cards_trajectory.append(cards_actions[j][i])

    return cards_trajectory, ddt


def main():
    db_path = os.path.join(db_dir, db_name) + ".db"
    con = sqlite3.connect(db_path)
    c = con.cursor()
    c.execute(f"""select * from records""")
    result = c.fetchall()
    num_deals = len(result)
    print(len(result))
    # print(result[0:10])
    cards = np.zeros([num_deals, NUM_CARDS], dtype=int)
    ddts = np.zeros([num_deals, 20], dtype=int)
    pars = np.zeros(num_deals, dtype=int)

    cards_, ddts_ = load_rl_dataset(db_name, flatten=True)
    for i in trange(num_deals):
        card, ddt = get_card_and_ddt_from_str(result[i][1])
        if not np.array_equal(cards_[i], card) or not np.array_equal(ddts_[i], ddt):
            print(i, " not equal!")
        cards[i] = card
        ddts[i] = ddt

    # print(cards)

    _, par_scores = calc_all_tables(cards)
    for i, par in enumerate(par_scores):
        par_score = get_par_score_from_par_results(par, view=0)
        pars[i] = par_score

    dataset = {
        "cards": cards,
        "ddts": ddts,
        "par_scores": pars
    }

    with open(os.path.join("../dataset/rl_data", f"{db_name}.pkl"), "wb") as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
