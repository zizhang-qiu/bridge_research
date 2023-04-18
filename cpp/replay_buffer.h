//
// Created by qzz on 2023/2/28.
//
#include <atomic>
#include "torch/torch.h"
#include "bridge_env.h"

#ifndef BRIDGE_RESEARCH_REPLAY_BUFFER_H
#define BRIDGE_RESEARCH_REPLAY_BUFFER_H
namespace rl::bridge {
class ReplayBuffer {
public:
    ReplayBuffer(int state_size,
                 int num_actions,
                 int capacity) :
            state_size_(state_size),
            num_actions_(num_actions),
            capacity_(capacity),
            cursor_(0),
            full_(false),
            size_(0),
            num_add_(0) {
        state_storage_ = torch::zeros({capacity_, state_size_}, {torch::kFloat});
        action_storage_ = torch::zeros(capacity_, {torch::kFloat});
        reward_storage_ = torch::zeros(capacity_, {torch::kFloat});
        log_probs_storage_ = torch::zeros({capacity_, num_actions_}, {torch::kFloat});
//        std::cout << state_storage_.sizes() << std::endl;
    };

    void Push(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &reward,
              const torch::Tensor &log_probs) {
        std::unique_lock<std::mutex> lk(m_);
//        std::cout << "push state" << std::endl;
        int cursor = cursor_.load();
        state_storage_[cursor] = state;
//        std::cout << "push action" << std::endl;
        action_storage_[cursor] = action;
//        std::cout << "push reward" << std::endl;
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

    int Size() {
        std::unique_lock<std::mutex> lk(m_);
        if (full_) {
            return capacity_;
        }
        return int(cursor_);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    Sample(int batch_size, const std::string &device) {
//        std::cout << "enter sample" << std::endl;
        std::unique_lock<std::mutex> lk(m_);
//        std::cout << "full: " << full_ << std::endl;
        int size = full_ ? capacity_ : cursor_.load();
//        std::cout << "size: " << size << std::endl;
        torch::Tensor indices = torch::multinomial(torch::ones(size), batch_size, false);
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
        std::cout << "full: " << full_ << std::endl;
        int size = full_ ? capacity_ : int(cursor_);
        std::cout << "size: " << size << std::endl;
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
    int size_;
    std::atomic<int> cursor_;
    std::atomic<bool> full_;
    std::atomic<int> num_add_;
    torch::Tensor state_storage_;
    torch::Tensor action_storage_;
    torch::Tensor reward_storage_;
    torch::Tensor log_probs_storage_;
    mutable std::mutex m_;
    std::mutex m_sampler_;
};
}
#endif //BRIDGE_RESEARCH_REPLAY_BUFFER_H
