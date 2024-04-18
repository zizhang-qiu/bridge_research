//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include <chrono>
#include <vector>
#include "rl/utils.h"
#include "rl/tensor_dict.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/cards_and_ddts.h"
#include "bridge_lib/bridge_card.h"
#include "sayc_bot.h"
#include "bridge_lib/dds_call.h"
#include "rl/context.h"
#include "bridge_thread_loop.h"
#include "rl/logger.h"
#include "bridge_lib/encoder.h"
using namespace rl;
using namespace rl::bridge;
struct Timer1 {
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();

  double Tick() {
    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};

int main() {
//  auto deal_manager = std::make_shared<BridgeDealManager>(cards, ddts);
//  auto deal = deal_manager->Next();
//  auto state = std::make_shared<BridgeBiddingState>(deal);
//  state->ApplyAction(0);
//  state->ApplyAction(6);
//  state->ApplyAction(0);
//  auto obs_tensor = state->ObservationTensor();
//  std::cout << obs_tensor.size() << std::endl;
//  rl::utils::PrintVector(obs_tensor);
//  rl::utils::PrintVector(rl::utils::GetNonZeroIndices(obs_tensor));
//  std::mt19937 rng(0);
//  std::vector<Cards> random_cards;
//  for (size_t i = 0; i < 2500000; ++i) {
//    random_cards.push_back(rl::utils::Permutation(0, kNumCards, rng));
//  }
//  auto cards_0 = rl::utils::Slice(cards, 0, 50);
//  auto cards_1 = rl::utils::Slice(cards, 50, 100);
//  auto ddts_0 = rl::utils::Slice(ddts, 0, 50);
//  auto ddts_1 = rl::utils::Slice(ddts, 50, 100);
//  Timer1 t;
//  auto deal_manager = std::make_shared<BridgeDealManager>(cards_0);
//  std::cout << t.Tick() << std::endl;
//  std::cout << deal_manager->Size() << std::endl;
//
//  // Build vec env
//  const std::vector<int> greedy{1, 1, 1, 1};
//  auto vec_env = std::make_shared<BridgeVecEnv>();
//  for (size_t i = 0; i < 50; ++i) {
//    const auto env = std::make_shared<BridgeBiddingEnv>(deal_manager, greedy);
//    vec_env->Push(env);
//  }
//  vec_env->Reset();
//  vec_env->Display(1);
//
//  deal_manager->Update(cards_1);
//  vec_env->ForceReset();
//  vec_env->Display(1);
//  rl::bridge::CanonicalObservationEncoder encoder;
//  auto hands_encoding = encoder.EncodeAllHands(state);
//  rl::utils::PrintVector(hands_encoding);
//  rl::utils::PrintVector(rl::utils::GetNonZeroIndices(hands_encoding));
//  auto history_encoding = encoder.EncodeHistory(state);
//  rl::utils::PrintVector(history_encoding);
//  rl::utils::PrintVector(rl::utils::GetNonZeroIndices(history_encoding));
  for(int rank=0; rank<13; ++rank){
    for(const Suit suit:kAllSuits){
      const BridgeCard card_value(suit, rank);
      std::cout << card_value.ToString() << ", " << card_value.Index() << std::endl;
    }
  }
  return 0;
}