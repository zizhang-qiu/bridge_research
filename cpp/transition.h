//
// Created by qzz on 2023/5/7.
//

#ifndef BRIDGE_RESEARCH_CPP_TRANSITION_H_
#define BRIDGE_RESEARCH_CPP_TRANSITION_H_
#include "rl/tensor_dict.h"
namespace rl::bridge {
class Transition {
 public:
  Transition() = default;

  Transition(TensorDict &obs,
             TensorDict &reply,
             torch::Tensor &reward,
             torch::Tensor &terminal,
             TensorDict &next_obs) :
      obs(obs),
      reply(reply),
      reward(reward),
      terminal(terminal),
      next_obs(next_obs) {
  }

  Transition Index(int i) const;

  static Transition MakeBatch(const std::vector<Transition>& transitions, const std::string& device);

  TensorDict obs;
  TensorDict reply;
  torch::Tensor reward;
  torch::Tensor terminal;
  TensorDict next_obs;
};
}
#endif //BRIDGE_RESEARCH_CPP_TRANSITION_H_
