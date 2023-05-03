//
// Created by qzz on 2023/5/3.
//
#include "multi_agent_transition_buffer.h"
#include "rl/logging.h"
namespace rl::bridge {

bool BridgeTransitionBuffer::PushObsActionLogProbs(const torch::Tensor &obs,
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

void BridgeTransitionBuffer::Clear() {
  obs_history_.clear();
  action_history_.clear();
  log_probs_history_.clear();
}

void BridgeTransitionBuffer::PushToReplayBuffer(std::shared_ptr<ReplayBuffer> &replay_buffer, double final_reward) {
  for (int i = obs_history_.size() - 1; i >= 0; --i) {
    replay_buffer->Push(obs_history_[i],
                        action_history_[i],
                        torch::tensor(final_reward),
                        log_probs_history_[i]);
  }
}

MultiAgentTransitionBuffer::MultiAgentTransitionBuffer(int num_agents) : num_agents_(num_agents) {
  for (size_t i = 0; i < num_agents_; i++) {
    storage_.emplace_back();
  }
}

void MultiAgentTransitionBuffer::PushObsActionLogProbs(int player,
                           const torch::Tensor &obs,
                           const torch::Tensor &action,
                           const torch::Tensor &log_probs) {
  storage_[player].PushObsActionLogProbs(obs, action, log_probs);
}

void MultiAgentTransitionBuffer::PushToReplayBuffer(std::shared_ptr<ReplayBuffer> replay_buffer, const std::vector<double> &reward) {
  RL_CHECK_EQ(reward.size(), num_agents_);
  for (size_t pl = 0; pl < num_agents_; ++pl) {
    storage_[pl].PushToReplayBuffer(replay_buffer, reward[pl]);
  }
}

void MultiAgentTransitionBuffer::Clear() {
  for (size_t pl = 0; pl < num_agents_; ++pl) {
    storage_[pl].Clear();
  }
}

}