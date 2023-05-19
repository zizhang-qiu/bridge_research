//
// Created by qzz on 2023/5/3.
//
#include "replay_buffer.h"

namespace rl::bridge {

ReplayBuffer::ReplayBuffer(int state_size,
                           int num_actions,
                           int capacity,
                           float alpha,
                           float eps,
                           float beta)
    : state_size_(state_size),
      num_actions_(num_actions),
      capacity_(capacity),
      alpha_(alpha),
      beta_(beta),
      eps_(eps),
      cursor_(0),
      full_(false),
      num_add_(0) {
  state_storage_ = torch::zeros({capacity_, state_size_}, {torch::kFloat});
  action_storage_ = torch::zeros(capacity_, {torch::kFloat});
  reward_storage_ = torch::zeros(capacity_, {torch::kFloat});
  log_probs_storage_ = torch::zeros({capacity_, num_actions_}, {torch::kFloat});
}

void ReplayBuffer::Push(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &reward,
                        const torch::Tensor &log_probs) {
  std::unique_lock<std::mutex> lk(m_);
  int cursor = cursor_.load();
  state_storage_[cursor] = state;
  action_storage_[cursor] = action;
  reward_storage_[cursor] = reward;
  log_probs_storage_[cursor] = log_probs;
  cursor_ = (cursor_ + 1) % capacity_;
  num_add_ += 1;
  if (!full_) {
    if (cursor_ == 0) {
      full_ = true;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::Sample(int batch_size, const std::string &device) {
  std::unique_lock<std::mutex> lk(m_);
  int size = full_ ? capacity_ : cursor_.load();
  torch::Tensor priorities = torch::abs(reward_storage_.slice(0, 0, size));
  torch::Tensor exponent_priorities = torch::pow(priorities, alpha_);
  torch::Tensor probs = exponent_priorities / torch::sum(exponent_priorities) + eps_;
//    torch::Tensor indices = torch::multinomial(torch::abs(reward_storage_.slice(0, 0, size)) + 0.05, batch_size, false);
  torch::Tensor indices = torch::multinomial(probs, batch_size, false);

  auto sample_states = state_storage_.index_select(0, indices);
  auto sample_actions = action_storage_.index_select(0, indices);
  auto sample_rewards = reward_storage_.index_select(0, indices);
  auto sample_log_probs = log_probs_storage_.index_select(0, indices);
  auto weights = priorities.index_select(0, indices);
  weights /= priorities.sum();
  weights = torch::pow(size * weights, -beta_);
  weights /= weights.max();
  if (device != "cpu") {
    auto d = torch::device(device);
    sample_states = sample_states.to(d);
    sample_actions = sample_actions.to(d);
    sample_rewards = sample_rewards.to(d);
    sample_log_probs = sample_log_probs.to(d);
    weights = weights.to(d);
  }
  return std::make_tuple(sample_states, sample_actions, sample_rewards, sample_log_probs, weights);
}


}