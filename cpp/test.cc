//
// Created by qzz on 2023/2/24.
//
#include <iostream>
#include "logging.h"
#include "tests/test_bridge_scoring.h"
#include "tests/cards_and_ddts.h"
#include "tests/test_bluechip_utils.h"
#include "tests/test_bridge_state.h"
#include "bridge_state.h"
#include "generate_deals.h"
#include "utils.h"
#include "search.h"
#include "torch/torch.h"

using namespace rl::utils;
using namespace rl::bridge;

int main() {
//  const torch::Tensor probs = torch::tensor({0.5, 0.3, 0.1, 0.1});
//  std::cout << probs << std::endl;
//  torch::Tensor top_k_indices, top_k_probs;
//  std::tie(top_k_indices, top_k_probs) = GetTopKActions(probs, 4, 0.1);
//  std::cout << "indices: " << top_k_indices << "\nprobs: " << top_k_probs << std::endl;
  torch::Tensor a = torch::tensor({1, 2, 3, 2});
  std::cout << torch::argmax(a, 0) << std::endl;
  return 0;
}