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
#include "generate_deals.h"
#include "utils.h"

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
//    std::mt19937 rng;
//    rng.seed(2);
//    auto cards_ = rl::utils::Permutation(0, kNumCards, rng);
  auto cards_ = cards[0];
  auto ddt = ddts[0];
//    auto holder = GetHolder(cards_);
//    auto ddt = CalcDDTable(holder);
//    std::vector<DDT> calc_ddts;
//    std::vector<int> calc_par_scores;
//    std::tie(calc_ddts, calc_par_scores) = CalcAllTablesOnce(cards);
//    RL_CHECK_EQ(calc_ddts, ddts);
//    rl::utils::PrintVector(calc_par_scores);
//    RL_CHECK_EQ(calc_par_scores, par_scores);
//    for(int i=0; i<cards.size();++i){
//
//    }
//    auto start_time = std::chrono::high_resolution_clock::now();
//    auto state = std::make_shared<BridgeBiddingState>(0, cards[0], false, false);
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//    std::cout << "Elapsed time: " << elapsed_time.count() << " microseconds" << std::endl;
//    std::cout << state << std::endl;
//
//    state->ApplyAction(0);
//    std::cout << state << std::endl;
////    std::cout << state->BidStr() << std::endl;
//    std::cout << state->ObservationString(0) << std::endl;
//    start_time = std::chrono::high_resolution_clock::now();
//    PrintVector(state->ObservationTensor());
//    end_time = std::chrono::high_resolution_clock::now();
//    elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//    std::cout << "Elapsed time: " << elapsed_time.count() << " microseconds" << std::endl;
//    state->ApplyAction(0);
//    state->ApplyAction(0);
//    state->ApplyAction(0);
//    std::cout << state << std::endl;
//    auto ddt_ = state->GetDoubleDummyTable();
//    PrintVector(ddt_);
//    RL_CHECK_EQ(ddt_, ddt);
//    std::cout << state->ObservationTensorSize() << std::endl;
//    return 0;
  BridgeDeal deal{cards_};
  std::cout << deal.ddt.has_value() << std::endl;
  std::cout << deal.par_score.has_value() << std::endl;
}