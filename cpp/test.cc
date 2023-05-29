//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include <chrono>
#include <vector>
#include "rl/utils.h"
#include "bridge_state.h"
#include "cards_and_ddts.h"
#include "sayc_bot.h"
#include "search.h"
#include "dds.h"
using namespace rl::bridge;
int main() {
//  auto deal_manager = rl::bridge::BridgeDealManager(cards, ddts, par_scores);
//  auto deal = deal_manager.Next();
//  auto state = std::make_shared<BridgeBiddingState>(deal);
//  auto hand_evaluation = state->GetHandEvaluation()[0];
//  rl::utils::PrintArray(hand_evaluation.hcp_per_suit);
//  std::cout << state->ObservationString(0) << std::endl;
//  auto obs_tensor = state->ObservationTensor2();
//  std::cout << obs_tensor.size() << std::endl;
//  rl::utils::PrintVector(obs_tensor);
//  rl::utils::PrintVector(rl::utils::GetNonZeroIndices(obs_tensor));
//  state->ApplyAction(6);
//  std::cout << state << std::endl;
//  obs_tensor = state->ObservationTensor2();
//  rl::utils::PrintVector(obs_tensor);
//  rl::utils::PrintVector(rl::utils::GetNonZeroIndices(obs_tensor));
////  std::cout << state << std::endl;
////  auto start = std::chrono::high_resolution_clock::now();
//  auto evaluations = state->GetHandEvaluation();
//  SAYCBot bot;
//  for(const auto e:evaluations){
//    std::cout << e.ToString() << std::endl;
//    std::cout << bot.CheckBalancedHand(e) << std::endl;
//  }
  for(const auto me:{kNorth, kEast, kSouth, kWest}){
    for(const auto target:{kNorth, kEast, kSouth, kWest}){
      std::cout << me << ", " << target << " : ";
      std::cout << RelativePlayer(me, target) << std::endl;
    }
  }
  return 0;
}