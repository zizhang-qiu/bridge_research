//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include <chrono>
#include <vector>
#include "rl/utils.h"
#include "rl/tensor_dict.h"
#include "bridge_state.h"
#include "cards_and_ddts.h"
#include "sayc_bot.h"
#include "dds.h"
#include "rl/context.h"
#include "bridge_thread_loop.h"
#include "rl/logger.h"
using namespace rl;
using namespace rl::bridge;
struct Timer {
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();

  double Tick() {
    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};

int main() {
//  auto deal_manager = std::make_shared<BridgeDealManager>(cards, ddts, par_scores, 150);
//  auto deal = deal_manager -> Next();
//  auto state = std::make_shared<BridgeBiddingState>(deal);
//  state->ApplyAction(6);
//  state->ApplyAction(0);
//  auto obs_tensor = state->ObservationTensor();
//  std::cout << obs_tensor.size() << std::endl;
//  rl::utils::PrintVector(obs_tensor);
//  rl::utils::PrintVector(rl::utils::GetNonZeroIndices(obs_tensor));
  auto card = rl::utils::Zeros<int>(20);
  rl::utils::PrintVector(card);

  return 0;
}