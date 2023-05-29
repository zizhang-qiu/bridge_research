//
// Created by qzz on 2023/5/21.
//
#include "dds.h"
#include "cards_and_ddts.h"
#include "gtest/gtest.h"
using namespace rl::bridge;

TEST(DDSTest, CalcDDTsTest){
  auto res = CalcDDTs(cards, -1);
  std::vector<std::vector<int>> calc_ddts(cards.size());
  for (int i = 0; i < cards.size(); ++i) {
    auto table_result = res[i];
    std::vector<int> ddt;

    for (const Denomination denomination : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      for (int pl = 0; pl < kNumPlayers; ++pl) {
        int t = table_result.resTable[DenominationToDDSStrain(denomination)][pl];
        ddt.emplace_back(t);
      }
    }
    calc_ddts[i] = ddt;
  }
  EXPECT_EQ(calc_ddts, ddts);
}