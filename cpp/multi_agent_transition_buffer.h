//
// Created by qzz on 2023/4/20.
//
#include "torch/torch.h"
#include "bridge_actor.h"
#include <vector>

#include "logging.h"

#ifndef BRIDGE_RESEARCH_CPP_MULTI_AGENT_TRANSITION_BUFFER_H_
#define BRIDGE_RESEARCH_CPP_MULTI_AGENT_TRANSITION_BUFFER_H_
namespace rl::bridge {


// A bridge transition buffer only stores
// obs, action and log probs, the sparse reward is given
// only at terminal as imp or duplicate score.
// So there is no need to store them all.
class BridgeTransitionBuffer {

  BridgeTransitionBuffer() = default;

  void PushObsActionLogProbs(const torch::Tensor &obs,
                             const torch::Tensor &action,
                             const torch::Tensor &log_probs) {
    RL_CHECK_EQ(obs_history_.size(), action_history_.size());
    RL_CHECK_EQ(action_history_.size(), log_probs_history_.size());
    obs_history_.emplace_back(obs);
    action_history_.emplace_back(action);
    log_probs_history_.emplace_back(log_probs);
  }

  void Clear() {
    obs_history_.clear();
    action_history_.clear();
    log_probs_history_.clear();
  }

 private:
  std::vector<torch::Tensor> obs_history_;
  std::vector<torch::Tensor> action_history_;
  std::vector<torch::Tensor> log_probs_history_;
};

class MultiAgentTransitionBuffer {
  explicit MultiAgentTransitionBuffer(int num_agents) : num_agents_(num_agents) {
    for (size_t i = 0; i < num_agents_; i++) {
      storage_.emplace_back(1.0);
    }
  }

  void PushObsAndActionAndLogProbs(int player,
                                   const torch::Tensor &obs,
                                   const torch::Tensor &action,
                                   const torch::Tensor &log_probs) {
    storage_[player].PushObsAndActionAndLogProbs(obs, action, log_probs);
  }

 private:
  int num_agents_;
  std::vector<TransitionBuffer> storage_;
};
}//namespace rl::bridge
#endif //BRIDGE_RESEARCH_CPP_MULTI_AGENT_TRANSITION_BUFFER_H_
