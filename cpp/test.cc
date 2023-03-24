//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include "logging.h"
#include "tests/test_bridge_scoring.h"
#include "tests/cards_and_ddts.h"
#include "tests/test_bluechip_utils.h"
#include "tests/test_imp_env.h"
#include "bridge_state.h"
#include "utils.h"
#include "absl/strings/str_format.h"
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>

using namespace rl::utils;
using namespace rl::bridge;



int main() {
//    TestScore();
//    std::mt19937 rng;
//    rng.seed(0);
//    TestReplayBuffer();
//    TestLoad();
//    TestThreadLoop();
//    WriteToDdt();
//    TestBlueChipUtils();
//    TestImpEnv();
    for (int i = 0; i < 5; ++i) {
        auto state = std::make_shared<BridgeBiddingState>(kNorth, cards[i], false, false, ddts[i]);
        auto ddt = state->GetDoubleDummyTable();
        RL_CHECK_EQ(ddt, ddts[i]);
        std::cout << ddt << std::endl;
    }

    return 0;
}