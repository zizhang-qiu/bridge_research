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
             TensorDict &next_obs)
      : obs(obs),
        reply(reply),
        reward(reward),
        terminal(terminal),
        next_obs(next_obs) {
  }

  [[nodiscard]] Transition Index(int i) const;

  static Transition MakeBatch(const std::vector<Transition> &transitions, const std::string &device);

  [[nodiscard]] TensorDict ToDict() const;

  [[nodiscard]] Transition SampleIllegalTransitions(float illegal_reward) const;

  // "s", "perfect_s", "legal_actions", "greedy"
  TensorDict obs;
  // "a", "log_probs", "values", "raw_probs"
  TensorDict reply;
  torch::Tensor reward;
  torch::Tensor terminal;
  TensorDict next_obs;
};

class SearchTransition {
 public:
  SearchTransition() = default;

  SearchTransition(TensorDict &obs,
                   torch::Tensor &policy_posterior,
                   torch::Tensor &value)
      : obs(obs),
        policy_posterior(policy_posterior),
        value(value) {}

  static SearchTransition MakeBatch(const std::vector<SearchTransition> &transitions, const std::string &device);

  TensorDict ToDict() const;

  TensorDict obs;
  torch::Tensor policy_posterior;
  torch::Tensor value;
};

class ObsBelief {
 public:
  ObsBelief() = default;

  ObsBelief(TensorDict &obs, TensorDict &belief)
      : obs(obs), belief(belief) {}

  [[nodiscard]] ObsBelief Index(int i) const;

  static ObsBelief MakeBatch(const std::vector<ObsBelief>& obs_beliefs, const std::string &device);
  TensorDict obs;
  TensorDict belief;
};
}
#endif //BRIDGE_RESEARCH_CPP_TRANSITION_H_
