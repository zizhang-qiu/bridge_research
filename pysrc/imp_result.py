# -*- coding:utf-8 -*-
# @FileName  :imp_result.py
# @Time      :2023/7/31 12:26
# @Author    :qzz
import pickle
from typing import List, Dict, Tuple

import numpy as np
import torch
import rl_cpp
from bridge_consts import (
    NUM_CARDS,
    NUM_PLAYERS,
    PBN_PREFIX,
    PBN_TEMPLATE_WITH_AUCTION,
    NUM_DENOMINATIONS,
    PLAYER_STR,
)
from pbn import cards_to_pbn


class IMPResult:
    cards: np.ndarray
    ddt: np.ndarray
    open_state = rl_cpp.BridgeBiddingState
    close_state = rl_cpp.BridgeBiddingState
    open_trajectory: List[int]
    close_trajectory: List[int]
    open_score: int
    close_score: int
    imp: int

    def __init__(
        self,
        cards: np.ndarray,
        ddt: np.ndarray,
        open_state: rl_cpp.BridgeBiddingState,
        close_state: rl_cpp.BridgeBiddingState,
    ):
        self.cards = cards
        self.ddt = ddt
        self.open_state = open_state
        self.close_state = close_state
        self.open_trajectory = self.open_state.history()
        self.close_trajectory = self.close_state.history()
        self.open_score = int(self.open_state.returns()[0])
        self.close_score = int(self.close_state.returns()[0])
        self.imp = rl_cpp.get_imp(self.open_score, self.close_score)

    def _open_state(self) -> rl_cpp.BridgeBiddingState:
        deal = rl_cpp.BridgeDeal()
        deal.cards = self.cards
        deal.ddt = self.ddt
        state = rl_cpp.BridgeBiddingState(deal)
        for action in self.open_trajectory[NUM_CARDS:]:
            state.apply_action(action)
        return state

    def _close_state(self) -> rl_cpp.BridgeBiddingState:
        deal = rl_cpp.BridgeDeal()
        deal.cards = self.cards
        deal.ddt = self.ddt
        state = rl_cpp.BridgeBiddingState(deal)
        for action in self.close_trajectory[NUM_CARDS:]:
            state.apply_action(action)
        return state

    def __repr__(self):
        msg = f"open:\n{self.open_state}\n\nclose:\n{self.close_state}\n\nimp: {self.imp}\n\n"
        return msg

    def __getstate__(self) -> Dict:
        spec = {
            "cards": self.cards,
            "ddt": self.ddt,
            "open_trajectory": self.open_trajectory,
            "close_trajectory": self.close_trajectory,
            "open_score": self.open_score,
            "close_score": self.close_score,
            "imp": self.imp,
        }
        return spec

    def __setstate__(self, state: Dict):
        self.cards = state["cards"]
        self.ddt = state["ddt"]
        self.open_trajectory = state["open_trajectory"]
        self.close_trajectory = state["close_trajectory"]
        self.open_score = state["open_score"]
        self.close_score = state["close_score"]
        self.imp = state["imp"]
        self.open_state = self._open_state()
        self.close_state = self._close_state()


def get_imp_list(res: List[IMPResult]):
    imp_list = []
    for imp_result in res:
        imp_list.append(imp_result.imp)
    return imp_list


def is_deal_same(res1: IMPResult, res2: IMPResult) -> bool:
    """
    Judge whether 2 IMPResults have same deals
    Args:
        res1: result1
        res2: result2

    Returns:
        bool: Whether the deals are same
    """
    return np.array_equal(res1.cards, res2.cards)


def count_win_draw_lose(imp_res: List[IMPResult]) -> Tuple[int, int, int]:
    win, draw, lose = 0, 0, 0
    for res in imp_res:
        if res.imp > 0:
            win += 1
        elif res.imp == 0:
            draw += 1
        else:
            lose += 1
    return win, draw, lose


def state_to_pbn(state: rl_cpp.BridgeBiddingState, imp: int) -> str:
    c = state.get_contract()
    contract_bid = (c.level - 1) * NUM_DENOMINATIONS + c.trumps()
    declarer_str = PLAYER_STR[c.declarer]
    bidding_history = state.bid_str_history()
    cards = state.history()[:NUM_CARDS]
    pbn_hand_string = cards_to_pbn(np.array(cards), 0)
    bidding_string = ""
    for i, bid_str in enumerate(bidding_history):
        bidding_string += bid_str
        if (i + 1) % NUM_PLAYERS == 0:
            bidding_string += "\n"
        else:
            bidding_string += " "
    # print(bidding_string)
    winning_partnership = "NS" if state.returns()[0] >= 0 else "EW"
    winning_score = int(abs(state.returns()[0]))
    score_str = (
        f"{winning_partnership}{' +' if winning_score > 0 else ' '}{winning_score}"
    )
    # pbn_str = PBN_TEMPLATE_WITH_AUCTION.format(dealer=0, deal=pbn_hand_string)
    tricks = state.num_declarer_tricks()
    contract_str = state.contract_str().split(" ")[0]
    pbn_str = PBN_TEMPLATE_WITH_AUCTION.format(
        dealer="N",
        deal=pbn_hand_string,
        declarer=declarer_str,
        contract=contract_str,
        results=tricks,
        comment="{" + f"IMP: {imp}" + "}",
        score=score_str,
        bidding_history=bidding_string,
    )
    return pbn_str


def imp_results_to_pbn(imp_results: List[IMPResult], save_path: str):
    pbn_str = PBN_PREFIX

    game_strs = []
    for res in imp_results:
        game_strs.append(state_to_pbn(res.open_state, res.imp))
        game_strs.append(state_to_pbn(res.close_state, res.imp))

    pbn_str += "\n\n".join(game_strs)

    with open(save_path, "w") as f:
        f.write(pbn_str)


# if __name__ == '__main__':
#     with open("imp_results_rl_bmcs.pkl", "rb") as fp:
#         imps: List[IMPResult] = pickle.load(fp)
#     w, d, l = count_win_draw_lose(imps)
#     print(w, d, l)
#
#     for imp_result in imps[:1]:
#         print(
#             f"{imp_result.open_state}\n\n{imp_result.open_state.returns()}\n\n{imp_result.close_state}"
#             f"\n\n{imp_result.close_state.returns()}\n\n{imp_result.imp}")
#     # with open("imp_results_rl.pkl", "rb") as fp:
#     #     imps = pickle.load(fp)
#     # w, d, l = count_win_draw_lose(imps)
#     # print(w, d, l)
#     imp_results_to_pbn(imps, "data/rl_bmcs.pbn")
