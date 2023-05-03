//
// Created by qzz on 2023/3/6.
//


#ifndef BRIDGE_RESEARCH_TEST_BLUECHIP_UTILS_H
#define BRIDGE_RESEARCH_TEST_BLUECHIP_UTILS_H
#include "bluechip_utils.h"
#include "gtest/gtest.h"
#endif //BRIDGE_RESEARCH_TEST_BLUECHIP_UTILS_H
namespace rl::bridge::bluechip {
const std::vector<std::string> AllBidStr = {
    "PASSES", "DOUBLES", "REDOUBLES", "1C", "1D", "1H", "1S", "1NT", "2C", "2D", "2H", "2S", "2NT",
    "3C", "3D", "3H", "3S", "3NT", "4C", "4D", "4H", "4S", "4NT", "5C", "5D", "5H", "5S", "5NT",
    "6C", "6D", "6H", "6S", "6NT", "7C", "7D", "7H", "7S", "7NT"
};

TEST(BluechipTest, TestBidActionToStr) {
  // bid action to str
  for (int bid = 0; bid < kNumCalls; ++bid) {
    auto bid_str = bluechip::BidActionToStr(bid);
    std::string expected_bid_str = bid > bluechip::kRedouble ? "bids " + AllBidStr[bid] : AllBidStr[bid];
    EXPECT_EQ(bid_str, expected_bid_str);
  }
}

TEST(BluechipTest, TestBidStrToAction) {
  std::vector<std::string> levels = {"1", "2", "3", "4", "5", "6", "7"};
  for (int i = bluechip::kRedouble + 1; i < kNumCalls; ++i) {
    int bid = bluechip::BidStrToAction(AllBidStr[i]);
    EXPECT_EQ(bid, i);
  }
}

TEST(BluechipTest, TestHandStrNoVoid) {
  // no void suit
  std::vector<int> cards = {47, 35, 31, 23, 11, 7, 50, 42, 34, 10, 6, 37, 20};
  auto hand_str = bluechip::GetHandString(cards);
  EXPECT_EQ(hand_str, "C 7. D J. H A Q T 4 3. S K T 9 7 4 3.");
}

TEST(BluechipTest, TestHandStrVoid) {
  // void suit
  std::vector<int> cards = {0, 1, 2, 4, 5, 6, 8, 9, 10, 13, 14, 20, 21};
  auto hand_str = bluechip::GetHandString(cards);
  EXPECT_EQ(hand_str, "C 7 4 3 2. D 7 5 4 3 2. H 5 4 3 2. S -.");
};
}