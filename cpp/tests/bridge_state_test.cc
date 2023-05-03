//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_TESTS_BRIDGE_STATE_TEST_CC_
#define BRIDGE_RESEARCH_CPP_TESTS_BRIDGE_STATE_TEST_CC_
#include "bridge_state.h"
#include "bridge_deal.h"
#include "cards_and_ddts.h"
#include "gtest/gtest.h"
using namespace rl::bridge;
TEST(BridgeStateTest, DDSTest) {
  for (size_t i = 0; i < 5; ++i) {
    BridgeDeal deal{cards[i]};
    auto state = std::make_shared<BridgeBiddingState>(deal);
    auto ddt = state->GetDoubleDummyTable();
    EXPECT_EQ(ddt, ddts[i]);
  }

}
#endif //BRIDGE_RESEARCH_CPP_TESTS_BRIDGE_STATE_TEST_CC_
