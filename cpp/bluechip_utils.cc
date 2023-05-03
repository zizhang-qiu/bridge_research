//
// Created by qzz on 2023/5/2.
//
#include "bluechip_utils.h"
namespace rl::bridge::bluechip{

Suit CardSuit(int card) { return Suit(card % kNumSuits); }

int CardRank(int card) { return card / kNumSuits; }

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
    auto suit = static_cast<int>(CardSuit(card));
    auto rank = CardRank(card);
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