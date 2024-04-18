//
// Created by qzz on 2023/6/19.
//
#include "bridge_utils.h"
namespace rl::bridge {
rl::bridge::ContractSampler::ContractSampler(int seed)
    : rng_(seed) {
  // All contract but passed out
  contract_index_dis = std::uniform_int_distribution<int>(1, kNumContracts - 1);
}
int ContractSampler::Sample() {
  int contract_index = contract_index_dis(rng_);
  return contract_index;
}
}

