//
// Created by qzz on 2023/3/4.
//


#ifndef BRIDGE_RESEARCH_RL_BASE_H
#define BRIDGE_RESEARCH_RL_BASE_H
#include "torch/torch.h"
#include "tensor_dict.h"
#include "types.h"
namespace rl {
class Env {
 public:
  Env() = default;

  ~Env() = default;

  virtual bool Reset() = 0;

  virtual void Step(const TensorDict &reply) = 0;

  [[nodiscard]] virtual std::vector<float> Returns() const = 0;

  [[nodiscard]] virtual TensorDict GetFeature() const = 0;

  [[nodiscard]] virtual bool Terminated() const = 0;

  [[nodiscard]] virtual Player CurrentPlayer() const = 0;
};

class Actor {
 public:
  Actor() = default;

  ~Actor() = default;

  virtual TensorDict Act(const TensorDict &obs) = 0;
};
}
#endif //BRIDGE_RESEARCH_RL_BASE_H
