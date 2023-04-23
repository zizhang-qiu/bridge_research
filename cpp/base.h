//
// Created by qzz on 2023/3/4.
//


#ifndef BRIDGE_RESEARCH_BASE_H
#define BRIDGE_RESEARCH_BASE_H
#include "torch/torch.h"
#include "tensor_dict.h"
namespace rl {
using ObsRewardTerminal = std::tuple<TensorDict, float, bool>;
class Env {
 public:
  Env() = default;

  ~Env() = default;

  virtual TensorDict Reset() = 0;

  virtual ObsRewardTerminal Step(const TensorDict &reply) = 0;

  virtual bool Terminated() const = 0;
};
}
#endif //BRIDGE_RESEARCH_BASE_H
