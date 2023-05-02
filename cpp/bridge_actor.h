//
// Created by qzz on 2023/2/23.
//


#ifndef BRIDGE_RESEARCH_BRIDGE_ACTOR_H
#define BRIDGE_RESEARCH_BRIDGE_ACTOR_H
#include "base.h"
#include "model_locker.h"
#include "replay_buffer.h"
#include "tensor_dict.h"
#include <utility>
using namespace torch::indexing;
namespace rl::bridge {
class RandomActor {
 public:
  RandomActor() = default;

  TensorDict Act(const torch::Tensor &obs) {
    auto legal_actions = obs.index({Slice(480, 518)});
    std::cout << legal_actions << std::endl;
    auto random_action = torch::multinomial(legal_actions, 1).squeeze();
    return {{"a", random_action}};
  }
};

class TransitionBuffer {
 public:
  explicit TransitionBuffer(float gamma) { gamma_ = gamma; }

  void PushObsAndActionAndLogProbs(const torch::Tensor &obs,
                                   const torch::Tensor &action,
                                   const torch::Tensor &log_probs) {
    obs_history_.emplace_back(obs);
    action_history_.emplace_back(action);
    log_probs_history_.emplace_back(log_probs);
  }

  void PushRewardAndTerminal(float reward, bool terminal) {
    reward_history_.emplace_back(reward);
    terminal_history_.emplace_back(int(terminal));
  }

  void PostToReplayBuffer(const std::shared_ptr<ReplayBuffer> &buffer,
                          float final_reward, bool clear = true) {
    //        std::vector<float> cum_reward =
    //        ComputeCumulativeRewards(final_reward);
    //        utils::PrintVector(cum_reward);
    for (int i = 0; i < obs_history_.size(); i++) {
      buffer->Push(obs_history_[i], action_history_[i],
                   torch::tensor(final_reward), log_probs_history_[i]);
    }
    if (clear) {
      Clear();
    }
  }

  void Clear() {
    obs_history_.clear();
    action_history_.clear();
    reward_history_.clear();
    terminal_history_.clear();
    log_probs_history_.clear();
  }

 private:
  std::vector<torch::Tensor> obs_history_;
  std::vector<torch::Tensor> action_history_;
  std::vector<torch::Tensor> log_probs_history_;
  std::vector<float> reward_history_;
  std::vector<int> terminal_history_;
  float gamma_;

  std::vector<float> ComputeCumulativeRewards(float final_reward) const {
    int reward_size = reward_history_.size();
    RL_CHECK_GT(reward_size, 0);
    //        std::cout << "reward size: " << reward_size << std::endl;
    std::vector<float> cum_reward(reward_size);
    cum_reward.back() = final_reward;
    if (reward_size > 1) {
      for (int i = reward_size - 2; i >= 0; --i) {
        cum_reward[i] = cum_reward[i + 1] * gamma_ + reward_history_[i];
      }
    }
    return cum_reward;
  }
};

class SingleEnvActor {

 public:
  explicit SingleEnvActor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(std::move(model_locker))) {
  };

  TensorDict Act(const TensorDict &obs) {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
    auto output = model.get_method("act")(input);
    auto reply = tensor_dict::FromIValue(output, torch::kCPU, true);

    model_locker_->ReleaseModel(id);
//    if (!eval_) {
//      transition_buffer_->PushObsAndActionAndLogProbs(obs, action, log_probs);
//    }
    return reply;
  }

  TensorDict GetTopKActionsWithMinProb(const TensorDict &obs, int k, float min_prob) {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
    torch::jit::IValue k_ = k;
    torch::jit::IValue min_prob_ = min_prob;
    input.emplace_back(k_);
    input.emplace_back(min_prob_);
    auto output = model.get_method("get_top_k_actions_with_min_prob")(input);
    auto reply = tensor_dict::FromIValue(output, torch::kCPU, true);
    model_locker_->ReleaseModel(id);
    return reply;
  }

  double GetProbForAction(const TensorDict &obs, Action action) {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
    torch::jit::IValue action_ = action;
    input.emplace_back(action_);
    auto output = model.get_method("get_prob_for_action")(input);
    auto reply = output.toDouble();

    model_locker_->ReleaseModel(id);
    return reply;
  }

//  void SetRewardAndTerminal(float reward, bool terminal) {
//    transition_buffer_->PushRewardAndTerminal(reward, terminal);
//  }
//
//  void PostToReplayBuffer(const std::shared_ptr<ReplayBuffer> &buffer,
//                          float final_reward) {
//    transition_buffer_->PostToReplayBuffer(buffer, final_reward, true);
//  }

 private:
  std::shared_ptr<ModelLocker> model_locker_;
//  int player_;
//  std::shared_ptr<TransitionBuffer> transition_buffer_;
//  bool eval_;
};

class VecEnvActor {
 public:
  explicit VecEnvActor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)) {};

  TensorDict Act(const TensorDict &obs) {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
    auto output = model.get_method("act")(input);
    auto reply = tensor_dict::FromIValue(output, torch::kCPU, true);
    model_locker_->ReleaseModel(id);
    return reply;
  }

 private:
  std::shared_ptr<ModelLocker> model_locker_;
};
} // namespace rl::bridge
#endif // BRIDGE_RESEARCH_BRIDGE_ACTOR_H
