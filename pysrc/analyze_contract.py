import pickle
from collections import Counter
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

import rl_cpp
from imp_result import IMPResult
import common_utils
import seaborn as sns
from bridge_consts import DENOMINATION_STR, LEVEL_STR, NUM_DENOMINATIONS


def get_heatmap_and_length(imp_results):
    contracts = []
    lengths = []
    for res in imp_results:
        open_state = res.open_state
        length = len(open_state.bid_history())
        lengths.append(length)
        contract = open_state.get_contract()
        if contract.declarer in [0, 2]:
            contracts.append(contract)

        close_state = res.close_state
        length = len(close_state.bid_history())
        lengths.append(length)
        contract = close_state.get_contract()
        if contract.declarer in [1, 3]:
            contracts.append(contract)
    # print(lengths)
    c = Counter(lengths)
    print(c)
    max_length = max(c.keys())
    lengths_count = np.array([c.get(i, 0) for i in range(max_length + 1)])
    # with open("data/length_rl_bmcs.pkl", "wb") as fp:
    #     pickle.dump(lengths_count, fp)
    print(lengths_count)

    print(len(contracts))
    contract_bids = [(c.level - 1) * NUM_DENOMINATIONS + c.trumps() for c in contracts]
    print(contract_bids)
    counter = Counter(contract_bids)
    counter = dict(counter)
    counter = sorted(counter.items(), key=lambda x: x[0])
    print(counter)
    with open("data/contract_rl.pkl", "wb") as fp:
        pickle.dump(counter, fp)

    heatmap_array = np.zeros(35, dtype=np.int32)

    for k, v in counter:
        heatmap_array[k] = v
    heatmap_array = np.reshape(heatmap_array, (7, 5))
    print(heatmap_array)

    return max_length, lengths_count, heatmap_array


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    i = 0
    for file in ["imp_results_rl.pkl", "imp_results_rl_bmcs.pkl"]:
        with open(file, "rb") as fp:
            imp_results: List[IMPResult] = pickle.load(fp)
        max_length, lengths_count, heatmap_array = get_heatmap_and_length(imp_results)
        # plt.figure()
        #
        axes[i].bar(np.arange(max_length + 1), lengths_count / lengths_count.sum())
        axes[i].set_xlabel("Bidding length", fontdict={"fontsize": 13})
        axes[i].set_ylabel("Frequency", fontdict={"fontsize": 13})
        # axes[i].set_xticks(ticks=np.arange(max_length), fontsize=13)
        # axes[i].set_yticks(np.arange(max_length), fontsize=13)
        axes[i].tick_params(axis="both", labelsize=12)
        axes[i].yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: "{}%".format(int(x * 100)))
        )
        # plt.show()

        club = "\u2663"
        diamond = "\u2666"
        heart = "\u2665"
        spade = "\u2660"
        suits = [club, diamond, heart, spade, "NT"]
        annot_array = [[f"{level + 1}{suit}" for suit in suits] for level in range(7)]
        sns.heatmap(
            (heatmap_array),
            # annot=annot_array,
            cmap="Blues",
            cbar=False,
            fmt="",
            xticklabels=False,
            yticklabels=False,
            annot_kws={"fontsize": 15},
            ax=axes[i],
        )
        suit_font = lambda suit: {
            "color": "red" if suit == diamond or suit == heart else "black",
            "weight": "normal",
            "size": 15,
        }
        for j, suit in enumerate(suits):
            for level in range(7):
                axes[i].text(
                    j + 0.3,
                    level + 0.55,
                    level + 1,
                    fontdict={"size": 15, "weight": "normal"},
                )
                axes[i].text(j + 0.5, level + 0.55, suit, fontdict=suit_font(suit))
        i += 1
    plt.show()
