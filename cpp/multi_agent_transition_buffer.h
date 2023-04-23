//
// Created by qzz on 2023/4/20.
//

#ifndef BRIDGE_RESEARCH_CPP_MULTI_AGENT_TRANSITION_BUFFER_H_
#define BRIDGE_RESEARCH_CPP_MULTI_AGENT_TRANSITION_BUFFER_H_
#include <vector>
#include "logging.h"
#include "utils.h"
#include "torch/torch.h"
#include "replay_buffer.h"
namespace rl::bridge {

// A bridge transition buffer only stores
// obs, action and log probs, the sparse reward is given
// only at terminal as imp or duplicate score.
// So there is no need to store them all.
class BridgeTransitionBuffer {
 public:
  BridgeTransitionBuffer() = default;

  bool PushObsActionLogProbs(const torch::Tensor &obs,
                             const torch::Tensor &action,
                             const torch::Tensor &log_probs) {
    RL_CHECK_EQ(obs_history_.size(), action_history_.size());
    RL_CHECK_EQ(action_history_.size(), log_probs_history_.size());
    bool available = utils::CheckProbNotZero(action, log_probs);
    if (available) {
      obs_history_.emplace_back(obs);
      action_history_.emplace_back(action);
      log_probs_history_.emplace_back(log_probs);
    }
    return available;
  }

  void Clear() {
    obs_history_.clear();
    action_history_.clear();
    log_probs_history_.clear();
  }

  void PushToReplayBuffer(std::shared_ptr<ReplayBuffer> &replay_buffer, double final_reward) {
    for (int i = obs_history_.size() - 1; i >= 0; --i) {
      replay_buffer->Push(obs_history_[i],
                          action_history_[i],
                          torch::tensor(final_reward),
                          log_probs_history_[i]);
    }
  }

 private:
  std::vector<torch::Tensor> obs_history_;
  std::vector<torch::Tensor> action_history_;
  std::vector<torch::Tensor> log_probs_history_;
};

class MultiAgentTransitionBuffer {
 public:

  explicit MultiAgentTransitionBuffer(int num_agents) : num_agents_(num_agents) {
    for (size_t i = 0; i < num_agents_; i++) {
      storage_.emplace_back();
    }
  }
  void PushObsActionLogProbs(int player,
                             const torch::Tensor &obs,
                             const torch::Tensor &action,
                             const torch::Tensor &log_probs) {
    storage_[player].PushObsActionLogProbs(obs, action, log_probs);
  }

  void PushToReplayBuffer(std::shared_ptr<ReplayBuffer> replay_buffer, const std::vector<double> &reward) {
    RL_CHECK_EQ(reward.size(), num_agents_);
    for (size_t pl = 0; pl < num_agents_; ++pl) {
      storage_[pl].PushToReplayBuffer(replay_buffer, reward[pl]);
    }
  }

  void Clear() {
    for (size_t pl = 0; pl < num_agents_; ++pl) {
      storage_[pl].Clear();
    }
  }

 private:
  int num_agents_;
  std::vector<BridgeTransitionBuffer> storage_;
};
}//namespace rl::bridge
#endif //BRIDGE_RESEARCH_CPP_MULTI_AGENT_TRANSITION_BUFFER_H_
