# -*- coding:utf-8 -*-
# @FileName  :analyze_opening_bid.py
# @Time      :2023/8/5 18:05
# @Author    :qzz
import pickle
from collections import Counter
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import common_utils
from bridge_consts import DENOMINATION_STR, LEVEL_STR
from imp_result import IMPResult


# class OpeningBidStat:
#     def __init__(self):
def bid_string(bid: int):
    if bid == 0:
        return "Pass"
    if bid == 1:
        return "Dbl"
    if bid == 2:
        return "RDbl"
    return LEVEL_STR[1 + (bid - 3) // 5] + DENOMINATION_STR[(bid - 3) % 5]


two_weak_length = 5


def judge_bal_hand(lengths: List[int]):
    sorted_lengths = sorted(lengths)
    return (
        sorted_lengths == [3, 3, 3, 4]
        or sorted_lengths == [2, 3, 4, 4]
        or sorted_lengths == [2, 3, 3, 5]
    )


def judge_1c(hcp: int, lengths: List[int]):
    return hcp >= 12 and lengths[0] >= 3


def judge_1d(hcp: int, lengths: List[int]):
    return hcp >= 12 and lengths[1] >= 3


def judge_1h(hcp: int, lengths: List[int]):
    return hcp >= 12 and lengths[2] >= 5


def judge_1s(hcp: int, lengths: List[int]):
    return hcp >= 12 and lengths[3] >= 5


def judge_1n(hcp: int, lengths: List[int]):
    return 15 <= hcp <= 17 and judge_bal_hand(lengths)


def judge_2c(hcp: int, lengths: List[int]):
    return hcp >= 22


def judge_2d(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and two_weak_length <= lengths[1] <= 7


def judge_2h(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and two_weak_length <= lengths[2] <= 7


def judge_2s(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and two_weak_length <= lengths[3] <= 7


def judge_2n(hcp: int, lengths: List[int]):
    return 20 <= hcp <= 21 and judge_bal_hand(lengths)


def judge_3c(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[0] == 7


def judge_3d(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[1] == 7


def judge_3h(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[2] == 7


def judge_3s(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[3] == 7


def judge_3n(hcp: int, lengths: List[int]):
    return 25 <= hcp <= 27 and judge_bal_hand(lengths)


def judge_4c(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[0] == 8


def judge_4d(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[1] == 8


def judge_4h(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[2] == 8


def judge_4s(hcp: int, lengths: List[int]):
    return 5 <= hcp <= 11 and lengths[3] == 8


judge_funcs: List[Callable[[int, List[int]], bool]] = [
    judge_1c,
    judge_1d,
    judge_1h,
    judge_1s,
    judge_1n,
    judge_2c,
    judge_2d,
    judge_2h,
    judge_2s,
    judge_2n,
    judge_3c,
    judge_3d,
    judge_3h,
    judge_3s,
    judge_3n,
    judge_4c,
    judge_4d,
    judge_4h,
    judge_4s,
]
num_funcs = len(judge_funcs)


def get_heatmap_array(imp_results):
    imps = []
    for res in imp_results:
        imps.append(res.imp)
    print(common_utils.get_avg_and_sem(imps))

    opening_bids = []
    hand_evaluations = {}
    opening_players = []

    opening_bids_wb5 = []
    hand_evaluations_wb5 = {}
    opening_players_wb5 = []

    num_passed_out = 0
    for res in imp_results:
        open_state = res.open_state
        bid, hand_evaluation, player = open_state.opening_bid_and_hand_evaluation()
        if bid != 0:
            if player in [0, 2]:
                opening_bids.append(bid)
                if bid not in hand_evaluations.keys():
                    hand_evaluations[bid] = [hand_evaluation]
                else:
                    hand_evaluations[bid].append(hand_evaluation)
            else:
                opening_bids_wb5.append(bid)
                if bid not in hand_evaluations_wb5.keys():
                    hand_evaluations_wb5[bid] = [hand_evaluation]
                else:
                    hand_evaluations_wb5[bid].append(hand_evaluation)
        else:
            num_passed_out += 1

        close_state = res.close_state
        bid, hand_evaluation, player = close_state.opening_bid_and_hand_evaluation()
        if bid != 0:
            if player in [1, 3]:
                opening_bids.append(bid)
                if bid not in hand_evaluations.keys():
                    hand_evaluations[bid] = [hand_evaluation]
                else:
                    hand_evaluations[bid].append(hand_evaluation)
            else:
                opening_bids_wb5.append(bid)
                if bid not in hand_evaluations_wb5.keys():
                    hand_evaluations_wb5[bid] = [hand_evaluation]
                else:
                    hand_evaluations_wb5[bid].append(hand_evaluation)
        else:
            num_passed_out += 1

    c = Counter(opening_bids)
    c = sorted(c.items(), key=lambda x: x[0])
    c = dict(c)
    print(c)

    heatmap_array = np.zeros((7, 5), dtype=np.int32)

    for i in range(3, 3 + 25):
        heatmap_array[(i - 3) // 5][(i - 3) % 5] = c.get(i, 0)
    print(heatmap_array)
    conformity = {}
    print(len(opening_bids))
    for key in sorted(hand_evaluations.keys()):
        # for key in [3]:
        evs = hand_evaluations[key]
        hcps = []
        club_lens = []
        diamond_lens = []
        heart_lens = []
        spades_lens = []
        conform = 0
        func_idx = key - 3
        for he in evs:
            if func_idx < num_funcs:
                conform += judge_funcs[func_idx](
                    he.high_card_points, he.length_per_suit
                )
            hcps.append(he.high_card_points)
            club_lens.append(he.length_per_suit[0])
            diamond_lens.append(he.length_per_suit[1])
            heart_lens.append(he.length_per_suit[2])
            spades_lens.append(he.length_per_suit[3])
        print(bid_string(key), len(evs), conform, conform / len(evs))
        if func_idx < num_funcs:
            conformity[key] = (len(evs), conform, conform / len(evs))
        # print(bid_string(key), np.mean(hcps), np.max(hcps), np.min(hcps))
        # print(np.mean(club_lens), np.mean(diamond_lens), np.mean(heart_lens), np.mean(spades_lens))
        # print(dict(Counter(hcps)))

    print(len(opening_bids_wb5))
    print(num_passed_out)
    conformity = {}
    for key in sorted(hand_evaluations_wb5.keys()):
        # for key in [3]:
        evs = hand_evaluations_wb5[key]
        hcps = []
        club_lens = []
        diamond_lens = []
        heart_lens = []
        spades_lens = []
        conform = 0
        func_idx = key - 3
        available = func_idx < num_funcs
        for he in evs:
            if available:
                conform += judge_funcs[func_idx](
                    he.high_card_points, he.length_per_suit
                )
            hcps.append(he.high_card_points)
            club_lens.append(he.length_per_suit[0])
            diamond_lens.append(he.length_per_suit[1])
            heart_lens.append(he.length_per_suit[2])
            spades_lens.append(he.length_per_suit[3])
        print(bid_string(key), len(evs), conform, conform / len(evs))
        if available:
            conformity[key] = (len(evs), conform, conform / len(evs))
    # with open("data/opening_bid_wbridge5.pkl", "wb") as fp:
    #     pickle.dump(conformity, fp)
    return heatmap_array


if __name__ == "__main__":
    # with open("imp_results_rl.pkl", "rb") as fp:
    #     imp_results: List[IMPResult] = pickle.load(fp)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    i = 0
    for file in ["imp_results_rl.pkl", "imp_results_rl_bmcs.pkl"]:
        with open(file, "rb") as fp:
            imp_results = pickle.load(fp)

        heatmap_array = get_heatmap_array(imp_results)

        club = "\u2663"
        diamond = "\u2666"
        heart = "\u2665"
        spade = "\u2660"
        suits = [club, diamond, heart, spade, "NT"]
        # print(club, diamond, heart, spade)
        annot_array = [[f"{level + 1}{suit}" for suit in suits] for level in range(7)]
        print(annot_array)
        # sns.set_style('whitegrid')
        h = sns.heatmap(
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
        # plt.text(0, 0, "12", ha="center", va="center", **suit_font)
        i += 1
    plt.show()
