//
// Created by qzz on 2023/2/28.
//

#ifndef BRIDGE_RESEARCH_REPLAY_BUFFER_H
#define BRIDGE_RESEARCH_REPLAY_BUFFER_H
#include <atomic>
#include "torch/torch.h"
namespace rl::bridge {
class ReplayBuffer {
 public:
  ReplayBuffer(int state_size, int num_actions, int capacity);

  void Push(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &reward,
            const torch::Tensor &log_probs);

  int Size() {
    std::unique_lock<std::mutex> lk(m_);
    if (full_) {
      return capacity_;
    }
    return int(cursor_);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  Sample(int batch_size, const std::string &device) {
    std::unique_lock<std::mutex> lk(m_);
    int size = full_ ? capacity_ : cursor_.load();
    torch::Tensor indices = torch::multinomial(torch::abs(reward_storage_.slice(0, 0, size)) + 0.05, batch_size, false);
//    torch::Tensor indices = torch::multinomial(torch::ones(size), batch_size, false);
    auto sample_states = state_storage_.index_select(0, indices);
    auto sample_actions = action_storage_.index_select(0, indices);
    auto sample_rewards = reward_storage_.index_select(0, indices);
    auto sample_log_probs = log_probs_storage_.index_select(0, indices);
    if (device != "cpu") {
      auto d = torch::device(device);
      sample_states = sample_states.to(d);
      sample_actions = sample_actions.to(d);
      sample_rewards = sample_rewards.to(d);
      sample_log_probs = sample_log_probs.to(d);
    }
    return std::make_tuple(sample_states, sample_actions, sample_rewards, sample_log_probs);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GetAll() {
    std::unique_lock<std::mutex> lk(m_);
    int size = full_ ? capacity_ : int(cursor_);
    auto states = state_storage_.slice(0, 0, size);
    auto actions = action_storage_.slice(0, 0, size);
    auto rewards = reward_storage_.slice(0, 0, size);
    auto log_probs = log_probs_storage_.slice(0, 0, size);
    return std::make_tuple(states, actions, rewards, log_probs);
  }

  int NumAdd() {
    return num_add_.load();
  }

 private:
  int state_size_;
  int num_actions_;
  int capacity_;
  std::atomic<int> cursor_;
  std::atomic<bool> full_;
  std::atomic<int> num_add_;
  torch::Tensor state_storage_;
  torch::Tensor action_storage_;
  torch::Tensor reward_storage_;
  torch::Tensor log_probs_storage_;
  mutable std::mutex m_;
};
}
#endif //BRIDGE_RESEARCH_REPLAY_BUFFER_H
