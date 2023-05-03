//
// Created by qzz on 2023/5/3.
//
#include "replay_buffer.h"

namespace rl::bridge {

ReplayBuffer::ReplayBuffer(int state_size,
                           int num_actions,
                           int capacity) :
    state_size_(state_size),
    num_actions_(num_actions),
    capacity_(capacity),
    cursor_(0),
    full_(false),
    num_add_(0) {
  state_storage_ = torch::zeros({capacity_, state_size_}, {torch::kFloat});
  action_storage_ = torch::zeros(capacity_, {torch::kFloat});
  reward_storage_ = torch::zeros(capacity_, {torch::kFloat});
  log_probs_storage_ = torch::zeros({capacity_, num_actions_}, {torch::kFloat});
};

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

}