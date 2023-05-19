//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include <chrono>
#include <vector>
#include "rl/utils.h"
#include "bridge_state.h"
#include "cards_and_ddts.h"
using namespace rl::bridge;
int main() {
  auto deal_manager = rl::bridge::BridgeDealManager(cards, ddts, par_scores);
  auto deal = deal_manager.Next();
  auto state = std::make_shared<BridgeBiddingState>(deal);
  std::cout << state << std::endl;
//  auto start = std::chrono::high_resolution_clock::now();
//  auto evaluations = state->GetHandEvaluation();
//  std::cout << evaluations[0].ToString() << std::endl;
//  auto end = std::chrono::high_resolution_clock::now();
//  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//  // Print the duration in microseconds
//  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
//
  std::vector<float> observation_tensor = state->ObservationTensorWithHandEvaluation();
  rl::utils::PrintVector(observation_tensor);
  auto indices = rl::utils::GetOneHotIndices(observation_tensor);
  rl::utils::PrintVector(indices);
  state->ApplyAction(6);
  observation_tensor = state->ObservationTensorWithHandEvaluation();
  rl::utils::PrintVector(observation_tensor);
  indices = rl::utils::GetOneHotIndices(observation_tensor);
  rl::utils::PrintVector(indices);
  std::cout << observation_tensor.size() << std::endl;
//  auto tokens = rl::utils::StrSplit("Hello,World,How,Are,You", ',');
//  rl::utils::PrintVector(tokens);
//  bool flag = rl::utils::StartsWith("Apple", "App");
//  std::cout << flag << std::endl;
//  flag = rl::utils::StrContains("Apple", "pl");
//  std::cout << flag << std::endl;
//  std::vector<std::string> elems = {"aa", "bb", "C"};
//  auto s = rl::utils::StrJoin(elems, ",");
//  std::cout << s << std::endl;
//  auto tokens = rl::utils::StrSplit(s, ',');
//  rl::utils::PrintVector(tokens);
//  auto s1 = rl::utils::AsciiStrToUpper(s);
//  std::cout << s1 << std::endl;
//  auto s2 = rl::utils::AsciiStrToLower(s);
//  std::cout << s2 << std::endl;

  return 0;
}