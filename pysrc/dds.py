#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:dds.py
@time:2023/01/15
"""
import math
import re
from ctypes import CDLL, Structure, c_int, c_uint, POINTER, byref, c_char, create_string_buffer
from enum import IntEnum
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from bridge_consts import NUM_CARDS, NUM_SUITS, NUM_CARDS_PER_SUIT, NUM_PLAYERS
from common_utils.assert_utils import assert_eq, assert_lteq

MAXNOOFBOARDS = 200
DDS_STRAINS = 5
DDS_HANDS = 4
DDS_SUITS = 4
R2 = 0x0004
R3 = 0x0008
R4 = 0x0010
R5 = 0x0020
R6 = 0x0040
R7 = 0x0080
R8 = 0x0100
R9 = 0x0200
RT = 0x0400
RJ = 0x0800
RQ = 0x1000
RK = 0x2000
RA = 0x4000

SolveBoardStatus = {
    1: "No fault",
    -1: "Unknown fault",
    -2: "Zero cards",
    -3: "Target > tricks left",
    -4: "Duplicated cards",
    -5: "Target < -1",
    -7: "Target > 13",
    -8: "Solutions < 1",
    -9: "Solutions > 3",
    -10: "> 52 cards",
    -12: "Invalid deal.currentTrick{Suit,Rank}",
    -13: "Card played in current trick is also remaining",
    -14: "Wrong number of remaining cards in a hand",
    -15: "threadIndex < 0 or >=noOfThreads, noOfThreads is the configured "
         "maximum number of threads"}

CALC_ALL_TABLES_BATCH_SIZE = 32


class ReturnCode(IntEnum):
    RETURN_NO_FAULT = 1
    RETURN_UNKNOWN_FAULT = -1
    RETURN_ZERO_CARDS = -2
    RETURN_TARGET_TOO_HIGH = -3
    RETURN_DUPLICATE_CARDS = -4
    RETURN_TARGET_WRONG_LO = -5
    RETURN_TARGET_WRONG_HI = -7
    RETURN_SOLNS_WRONG_LO = -8
    RETURN_SOLNS_WRONG_HI = -9
    RETURN_TOO_MANY_CARDS = -10
    RETURN_SUIT_OR_RANK = -12
    RETURN_PLAYED_CARD = -13
    RETURN_CARD_COUNT = -14
    RETURN_THREAD_INDEX = -15
    RETURN_MODE_WRONG_LO = -16
    RETURN_MODE_WRONG_HI = -17
    RETURN_TRUMP_WRONG = -18
    RETURN_FIRST_WRONG = -19


class DDTabelDeal(Structure):
    _fields_ = [("cards", c_uint * 4 * 4)]


class DDTableDeals(Structure):
    _fields_ = [("noOfTables", c_int),
                ("deals", DDTabelDeal * (MAXNOOFBOARDS * DDS_STRAINS))]


class DDTableResults(Structure):
    _fields_ = [("resTable", c_uint * 4 * 5)]


class DDTableRes(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("results", DDTableResults * (MAXNOOFBOARDS * DDS_STRAINS))]


class ParResults(Structure):
    _fields_ = [("parScore", ((c_char * 16) * 2)),
                ("parContractsString", ((c_char * 128) * 2))]


class AllParResults(Structure):
    _fields_ = [("presults", ParResults * MAXNOOFBOARDS)]


class ParResultsDealer(Structure):
    _fields_ = [("number", c_int),
                ("score", c_int),
                ("contracts", c_char * 10 * 10)]


dll = CDLL("D:/Projects/bridge_research/pysrc/dds.dll", winmode=0)
dll.CalcDDtable.argtypes = (DDTabelDeal, POINTER(DDTableResults))
dll.CalcDDtable.restype = c_int
dll.CalcAllTables.argtypes = (POINTER(DDTableDeals), c_int, c_int * DDS_STRAINS, POINTER(DDTableRes),
                              POINTER(AllParResults))
dll.CalcAllTables.restype = c_int
dll.Par.argtypes = (POINTER(DDTableResults), POINTER(ParResults), c_int)
dll.Par.restype = c_int


def deal_return_code(return_code: int):
    if return_code != ReturnCode.RETURN_NO_FAULT:
        line = create_string_buffer(80)
        dll.ErrorMessage(return_code, line)
        print(f"DDS error: {line.value.decode('utf-8')}")
        raise ValueError


def holder_to_dd_table_deal(holder: np.ndarray) -> DDTabelDeal:
    """
    Convert a holder to a dd table deal object.
    Args:
        holder: A arrays tracks holder for each card.

    Returns:
        The dd table deal.
    """
    dd_table_deal = DDTabelDeal()
    for suit in range(NUM_SUITS):
        for rank in range(NUM_CARDS_PER_SUIT):
            player = holder[suit + rank * NUM_SUITS]
            dd_table_deal.cards[player][3 - suit] += 1 << (2 + rank)
    return dd_table_deal


def dds_ddt_to_bridge_ddt(dds_ddt: np.ndarray) -> np.ndarray:
    """
    Convert the ddt given by double dummy solver to out representation.
    The ddt given by dds is ordered S,H,D,C,NT, our order is C,D,H,S,NT.
    Args:
        dds_ddt: The ddt given by dds.

    Returns:
        The correct ddt.
    """
    assert np.array_equal(dds_ddt.shape, (DDS_STRAINS, DDS_HANDS))
    no_trump_table = dds_ddt[4:]
    trump_table = np.flip(dds_ddt[0:4], 0)
    return np.vstack([trump_table, no_trump_table])


def calc_dd_table(holder: np.ndarray) -> DDTableResults:
    """
    Calculate ddt for one table.
    Args:
        holder: The holder of each card.

    Returns:
        The ddt.
    """
    assert_eq(len(holder), NUM_CARDS)
    dd_table_deal = holder_to_dd_table_deal(holder)
    dd_table_results = DDTableResults()
    dll.SetMaxThreads(0)
    return_code = dll.CalcDDtable(dd_table_deal, byref(dd_table_results))
    deal_return_code(return_code)
    return dd_table_results


def dd_table_results_to_ddt(dd_table_results: DDTableResults):
    return dds_ddt_to_bridge_ddt(np.ctypeslib.as_array(dd_table_results.resTable)).flatten()


def get_holder_from_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """
    Get holder of each card from a card trajectory.
    Args:
        trajectory: The card trajectory.

    Returns:
        A numpy array indicates the holder of each card.
    """
    assert_eq(trajectory.size, NUM_CARDS)
    holder = np.full(NUM_CARDS, -1)
    for length, card in enumerate(trajectory):
        holder[card] = length % NUM_PLAYERS
    return holder


def _calc_all_tables_once(trajectories: np.ndarray) -> Tuple[DDTableRes, AllParResults]:
    """
    Since calc all table can only calc 32 tables, this function is used for calc once
    Args:
        trajectories: the trajectories for bridge, should be less than 32 in shape[0]
    Returns:
        the ddts in numpy array form and par scores
    """
    assert_eq(trajectories.ndim, 2)
    assert_lteq(trajectories.shape[0], CALC_ALL_TABLES_BATCH_SIZE)
    num_batch_deals = trajectories.shape[0]
    holders = np.full_like(trajectories, fill_value=-1)
    for i, trajectory in enumerate(trajectories):
        holder = get_holder_from_trajectory(trajectory)
        holders[i] = holder
    dd_table_deals = DDTableDeals()
    dd_table_deals.noOfTables = num_batch_deals
    dd_table_res = DDTableRes()

    # write into dd_table_deals
    for i in range(num_batch_deals):
        # print(i)
        dd_table_deal = holder_to_dd_table_deal(holders[i])
        dd_table_deals.deals[i] = dd_table_deal

    pres = AllParResults()
    mode = 0
    c_trump_filter = (c_int * DDS_STRAINS)(0, 0, 0, 0, 0)
    return_code = dll.CalcAllTables(byref(dd_table_deals), mode, c_trump_filter, byref(dd_table_res), byref(pres))
    deal_return_code(return_code)
    # ddts = [dds_ddt_to_bridge_ddt(np.ctypeslib.as_array(dd_table_res.results[j].resTable)) for j in
    #         range(num_batch_deals)]
    # p_result = pres.presults
    # par_scores = [p_result[j] for j in range(num_batch_deals)]

    return dd_table_res, pres


def calc_all_tables(trajectories: np.ndarray, show_progress_bar=True) -> Tuple[List[DDTableRes], List[AllParResults]]:
    """
    Calculate all tables of given card trajectories.
    Args:
        trajectories: The card trajectories.
        show_progress_bar: Whether to show the progress bar.

    Returns:
        The ddts and par scores
    """
    assert_eq(trajectories.ndim, 2)
    assert_eq(trajectories.shape[1], NUM_CARDS)
    num_deals = trajectories.shape[0]
    num_batches = math.ceil(num_deals / CALC_ALL_TABLES_BATCH_SIZE)
    dd_table_res_list = []
    pres_list = []
    for i_batch in tqdm(range(num_batches), disable=not show_progress_bar):
        left = i_batch * CALC_ALL_TABLES_BATCH_SIZE
        right = min((i_batch + 1) * CALC_ALL_TABLES_BATCH_SIZE, num_deals)
        batch_trajectories = trajectories[left:right]
        dd_table_res, pres = _calc_all_tables_once(batch_trajectories)
        dd_table_res_list.append(dd_table_res)
        pres_list.append(pres)

    return dd_table_res_list, pres_list


def get_ddts_from_dd_table_res_list(dd_table_res_list: List[DDTableRes]):
    res = []
    for dd_table_res in dd_table_res_list:
        num_boards = dd_table_res.noOfBoards // 20
        for j in range(num_boards):
            dd_table_results = dd_table_res.results[j]
            res.append(dd_table_results_to_ddt(dd_table_results))
    return np.array(res, dtype=int)


def get_par_score_and_contract_from_par_results(par_results: ParResults) -> Tuple[List[int], List[str]]:
    """
    Get the par score from par results
    Args:
        par_results: The par results

    Returns:
        The par score.
    """
    par_scores = []
    par_contracts = []
    for view in [0, 1]:
        par_score_str = par_results.parScore[view].value.decode("utf-8")
        par_contract_str = par_results.parContractsString[view].value.decode("utf-8")
        par_score = int(re.search(r"[-]?\d+", par_score_str).group())
        par_scores.append(par_score)
        par_contracts.append(par_contract_str)
    return par_scores, par_contracts


def get_par_scores_and_contracts_from_pres_list(pres_list: List[AllParResults]) \
        -> Tuple[List[List[int]], List[List[str]]]:
    par_scores_list = []
    par_contracts_list = []
    for pres in pres_list:
        presults = pres.presults
        for i in range(len(presults)):
            par_results = presults[i]
            if not par_results.parScore[0].value.decode("utf-8"):
                break
            par_scores, par_contracts = get_par_score_and_contract_from_par_results(par_results)
            # print(par_scores, par_contracts)
            par_scores_list.append(par_scores)
            par_contracts_list.append(par_contracts)
    return par_scores_list, par_contracts_list

# holdings = [
#     [
#         [RQ | RJ | R6, R8 | R7 | R3, RK | R5, RA | RT | R9 | R4 | R2],
#         [RK | R6 | R5 | R2, RJ | R9 | R7, RT | R8 | R3, RA | RQ | R4],
#         [RJ | R8 | R5, RA | RT | R7 | R6 | R4, RK | RQ | R9, R3 | R2],
#         [RT | R9 | R8, RQ | R4, RA | R7 | R6 | R5 | R2, RK | RJ | R3]
#     ],
#     [
#         [RA | RK | R9 | R6, RQ | RJ | RT | R5 | R4 | R3 | R2, 0, R8 | R7],
#         [RK | RQ | R8, RT, RJ | R9 | R7 | R5 | R4 | R3, RA | R6 | R2],
#         [RA | R9 | R8, R6, RK | R7 | R5 | R3 | R2, RQ | RJ | RT | R4],
#         [RK | R6 | R3, RQ | RJ | R8 | R2, R9 | R4, RA | RT | R7 | R5]
#     ],
#     [
#         [R7 | R3, RQ | RT | R6, R5, RA | RK | RJ | R9 | R8 | R4 | R2],
#         [RQ | RJ | RT, R8 | R7 | R6, RA | R9 | R5 | R4 | R3 | R2, RK],
#         [RA | RQ | R5 | R4, RK | RJ | R9, R7 | R6 | R3 | R2, RT | R8],
#         [RT | R7 | R5 | R2, RA | RQ | R8 | R4, RK | R6, RJ | R9 | R3]
#     ]
# ]
