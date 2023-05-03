//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include <vector>
#include "rl/utils.h"
#include "bridge_scoring.h"

int main() {
//  std::cout << rl::bridge::GetImp(100, 50) << std::endl;
  auto all_scores = rl::bridge::AllScores();
  std::cout << all_scores.size() << std::endl;
  rl::utils::PrintVector(all_scores);
//  for (const auto score : all_scores) {
//    std::cout << score << std::endl;
//  }
  return 0;
}