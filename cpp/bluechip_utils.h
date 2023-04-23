//
// Created by qzz on 2023/3/6.
//


#ifndef BRIDGE_RESEARCH_BLUECHIP_UTILS_H
#define BRIDGE_RESEARCH_BLUECHIP_UTILS_H
#include "logging.h"
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

int BidStrToAction(const std::string &bid_str) {
    int level = std::stoi(bid_str.substr(0, 1));
    int trump = std::find(Denominations.begin(), Denominations.end(), bid_str.substr(1)) - Denominations.begin();
    int action = kActionBid + (level - 1) * kNumDenominations + trump;
    RL_CHECK_GE(action, kActionBid);
    RL_CHECK_LT(action, kNumCalls);
    return action;
}

std::string BidActionToStr(const int bid) {
    RL_CHECK_GE(bid, kPass);
    RL_CHECK_LT(bid, kNumCalls);
    switch (bid) {
        case kPass:
            return "PASSES";
        case kDouble:
            return "DOUBLES";
        case kRedouble:
            return "REDOUBLES";
        default:
            std::string level = std::to_string((bid - kActionBid) / kNumDenominations + 1);
            const std::string &trump = Denominations[(bid - kActionBid) % kNumDenominations];
            return "bids " + level + trump;
    }
}

std::string GetHandString(std::vector<int> cards) {
    RL_CHECK_EQ(cards.size(), kNumCardsPerHand);
    std::array<std::vector<std::string>, kNumPlayers> suits;
    std::vector<std::string> suits_str(kNumSuits);
    std::sort(cards.begin(), cards.end(), std::greater<>());
    for (auto card: cards) {
        auto suit = static_cast<int>(bridge::CardSuit(card));
        auto rank = bridge::CardRank(card);
        suits[suit].emplace_back(Ranks[rank]);
    }

    for (int i = 0; i < kNumSuits; i++) {
        if (!suits[i].empty()) {
            suits_str[i] = Suits[i] + " " + utils::StrJoin(suits[i], " ") + ".";
        } else {
            suits_str[i] = Suits[i] + " -.";
        }
    }
    std::string ret = utils::StrJoin(suits_str, " ");
    return ret;
}

}
#endif //BRIDGE_RESEARCH_BLUECHIP_UTILS_H