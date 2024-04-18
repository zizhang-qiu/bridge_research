//
// Created by qzz on 2023/3/6.
//


#ifndef BRIDGE_RESEARCH_BLUECHIP_UTILS_H
#define BRIDGE_RESEARCH_BLUECHIP_UTILS_H
#include "rl/logging.h"
#include "bridge_state.h"
#include "str_utils.h"
#include <string>
#include <vector>
#include <array>

namespace rl::bridge::bluechip {
//BlueChip bridge protocol message constants
inline constexpr int kPass = 0;
inline constexpr int kDouble = 1;
inline constexpr int kRedouble = 2;
inline constexpr int kActionBid = 3;
const std::vector<std::string> Seats = {"NORTH", "EAST", "SOUTH", "WEST"};
const std::vector<std::string> Denominations = {"C", "D", "H", "S", "NT"};
const std::vector<std::string> Suits = {"C", "D", "H", "S"};
const std::vector<std::string> Ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"};

int BidStrToAction(const std::string &bid_str);

std::string BidActionToStr(int bid);

Suit CardSuit(int card);

int CardRank(int card);

std::string GetHandString(std::vector<int> cards);

}
#endif //BRIDGE_RESEARCH_BLUECHIP_UTILS_H