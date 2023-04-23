//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_TESTS_TEST_BRIDGE_STATE_H_
#define BRIDGE_RESEARCH_CPP_TESTS_TEST_BRIDGE_STATE_H_
#include "../bridge_state.h"
#include "../bridge_deal.h"
#include "cards_and_ddts.h"
namespace rl::bridge {
void TestDDS() {
  for (size_t i = 0; i < 5; ++i) {
    BridgeDeal deal{cards[i]};
    auto state = std::make_shared<BridgeBiddingState>(deal);
//      std::cout << state << std::endl;
    auto ddt = state->GetDoubleDummyTable();
    RL_CHECK_EQ(ddt, ddts[i]);
  }
  std::cout << "Pass dds test." << std::endl;
}
}
#endif //BRIDGE_RESEARCH_CPP_TESTS_TEST_BRIDGE_STATE_H_
