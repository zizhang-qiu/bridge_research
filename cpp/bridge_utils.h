//
// Created by qzz on 2023/6/19.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_UTILS_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_UTILS_H_
#include <random>
#include "bridge_scoring.h"
namespace rl::bridge{
class ContractSampler{
 public:
  explicit ContractSampler(int seed);

  int Sample();
 private:
  std::mt19937 rng_;
  std::uniform_int_distribution<int> contract_index_dis;
};
}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_UTILS_H_
