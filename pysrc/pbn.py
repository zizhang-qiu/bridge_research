#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:pbn.py
@time:2023/02/17
"""
import os.path
import re
from typing import Tuple, Union, List

import numpy as np
from tqdm import tqdm

from pysrc.bridge_vars import Suit, RANK_STR, NUM_CARDS_PER_HAND, NUM_PLAYERS, NUM_CARDS, NUM_DENOMINATIONS, \
    Denomination, NUM_SUITS, PLAYER_STR, PBN_TEMPLATE, PBN_PREFIX
from dds import DDS_STRAINS, DDS_HANDS
from common_utils.assert_utils import assert_eq


def get_card(suit: int, rank: int) -> int:
    return rank * NUM_SUITS + suit


def get_suit(card: int) -> int:
    return card % NUM_SUITS


def get_rank(card: int) -> int:
    return card // NUM_SUITS


def ddt_char_2_int(ddt_char: str) -> int:
    """
    function to convert ddt character to int, ddt in a pbn file will use 'a' to represent 10, 'b' as 11, etc.

    Args:
        ddt_char: (str)the character

    Returns:the int result

    """
    assert len(ddt_char) == 1, "only one character could be input"
    if ddt_char.isdigit():
        return int(ddt_char)
    else:
        return ord(ddt_char) - ord("a") + 10


def get_trajectories_and_ddts_from_pbn_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get cards and ddts from a pbn file
    Args:
        file_path: The path to the pbn file

    Returns:
        The numpy array of cards and ddts
    """
    with open(file_path, "r") as f:
        content = f.read()
    # print(content)

    games = content.split('[Event ""]')[1:]
    num_deals = len(games)
    cards = np.zeros([num_deals, NUM_CARDS], dtype=int)
    ddts = np.full([num_deals, DDS_STRAINS, DDS_HANDS], dtype=int, fill_value=-1)

    for i_game, game in enumerate(games):
        match = re.search('\\"(.):(.*?)\"', game)
        cards_list = match.group(2).split(' ')
        dealer = match.group(1)
        # print(cards)
        cards_actions = []
        for card in cards_list:
            card_per_suit = card.split('.')
            cards_action = []
            for suit, ranks in enumerate(card_per_suit):
                for rank in ranks:
                    cards_action.append(get_card(3 - Suit(suit), RANK_STR.index(rank)))
            cards_actions.append(cards_action)

        cards_trajectory = []
        for i in range(NUM_CARDS_PER_HAND):
            for j in range(NUM_PLAYERS):
                cards_trajectory.append(cards_actions[j][i])
        cards[i_game] = np.roll(cards_trajectory, PLAYER_STR.index(dealer))

        ddt_str = re.search('DoubleDummyTricks "(.*?)"]', game)
        if ddt_str:
            ddt_str = ddt_str.group(1)
            # the pbn ddt in ordered N(NSHDC)-S-E-W
            ddt_list = [ddt_char_2_int(ddt_char) for ddt_char in ddt_str]
            ddt_list[5:10], ddt_list[10:15] = ddt_list[10:15], ddt_list[5:10]
            ddt = np.full([DDS_STRAINS, DDS_HANDS], dtype=int, fill_value=-1)

            for trump in Denomination:
                for player in range(NUM_PLAYERS):
                    ddt[trump][player] = ddt_list[NUM_DENOMINATIONS - trump - 1 + player * NUM_DENOMINATIONS]
            ddts[i_game] = ddt

    return cards, ddts


def _cards_pbn_format(cards: np.ndarray):
    assert_eq(cards.size, NUM_CARDS_PER_HAND)
    suits = [[] for _ in range(NUM_SUITS)]
    suits_str = ["" for _ in range(NUM_SUITS)]
    sorted_cards = sorted(cards, reverse=True)
    for card in sorted_cards:
        suit = get_suit(card)
        rank = get_rank(card)
        suits[suit].append(RANK_STR[rank])

    for i in range(NUM_SUITS):
        if suits[i]:
            suits_str[i] = "".join(suits[i])

    ret = ".".join(reversed(suits_str)).rstrip(".")
    return ret


def cards_to_pbn(cards: np.ndarray, dealer: Union[str, int]) -> str:
    """
    Convert cards actions to pbn format
    Args:
        dealer: The dealer
        cards: The cards array

    Returns:
        The hand string of pbn format
    """
    assert_eq(cards.size, NUM_CARDS)
    hand_strs = []
    for i in range(NUM_PLAYERS):
        cards_this_player = cards[i::NUM_PLAYERS]
        # print(cards_this_player)
        this_hand_string = _cards_pbn_format(cards_this_player)
        # print(this_hand_string)
        hand_strs.append(this_hand_string)
    pbn_hand_string = " ".join(hand_strs)
    # print(pbn_hand_string)
    if isinstance(dealer, int):
        dealer = PLAYER_STR[dealer]

    pbn_hand_string = f"{dealer}:{pbn_hand_string}"
    return pbn_hand_string


def get_pbn_game_string(hand_string: str, dealer: Union[int, str]):
    """
    Get the string of pbn format.
    Args:
        hand_string: The string of hand
        dealer: The dealer.

    Returns:
        The string.
    """
    if isinstance(dealer, int):
        dealer = PLAYER_STR[dealer]
    res = PBN_TEMPLATE.format(dealer=dealer, deal=hand_string)
    return res


def create_pbn_file(file_name: str, file_dir: str, game_strs: List[str]):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    if not file_name.endswith(".pbn"):
        file_name += ".pbn"
    file_path = os.path.join(file_dir, file_name)
    pbn_str = PBN_PREFIX
    pbn_str += "\n\n".join(game_strs)

    with open(file_path, "w") as f:
        f.write(pbn_str)


def write_pbn_file(cards: np.ndarray, file_name: str, file_dir: str):
    assert_eq(cards.ndim, 2)
    game_strs = []
    for c in tqdm(cards):
        game_str = get_pbn_game_string(cards_to_pbn(c, 0), 0)
        game_strs.append(game_str)

    create_pbn_file(file_name, file_dir, game_strs)


if __name__ == '__main__':
    pass
