//
// Created by qzz on 2023/2/23.
//
#include "base.h"
#include "model_locker.h"
#include "replay_buffer.h"
#include <utility>

#ifndef BRIDGE_RESEARCH_BRIDGE_ACTOR_H
#define BRIDGE_RESEARCH_BRIDGE_ACTOR_H
using namespace torch::indexing;
namespace rl::bridge {
class RandomActor : public Actor {
public:
  RandomActor() = default;

  torch::Tensor Act(const torch::Tensor &obs) override {
    auto legal_actions = obs.index({Slice(480, 518)});
    std::cout << legal_actions << std::endl;
    auto random_action = torch::multinomial(legal_actions, 1);
    return random_action;
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

class SingleEnvActor : public Actor {

public:
  SingleEnvActor(std::shared_ptr<ModelLocker> model_locker, int player,
                 float gamma, bool eval)
      : model_locker_(std::move(std::move(model_locker))), player_(player),
        eval_(eval) {
    transition_buffer_ = std::make_shared<TransitionBuffer>(gamma);
  };

  torch::Tensor Act(const torch::Tensor &obs) override {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(obs.to(model_locker_->device_));
    auto output = model.get_method("act")(input);
    auto out_tuple = output.toTuple();
    auto action = out_tuple->elements()[0].toTensor();
    //        std::cout << action << std::endl;
    auto log_probs = out_tuple->elements()[1].toTensor();
    //        std::cout << log_probs << std::endl;
    auto available = out_tuple->elements()[2].toBool();

    model_locker_->ReleaseModel(id);
    if (!eval_) {
      if (available) {
        transition_buffer_->PushObsAndActionAndLogProbs(obs, action, log_probs);
      }
    }
    return action;
  }

  void SetRewardAndTerminal(float reward, bool terminal) {
    transition_buffer_->PushRewardAndTerminal(reward, terminal);
  }

  void PostToReplayBuffer(const std::shared_ptr<ReplayBuffer> &buffer,
                          float final_reward) {
    transition_buffer_->PostToReplayBuffer(buffer, final_reward);
  }

private:
  std::shared_ptr<ModelLocker> model_locker_;
  int player_;
  std::shared_ptr<TransitionBuffer> transition_buffer_;
  bool eval_;
};

class VecEnvActor : public Actor {
public:
  VecEnvActor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)){};

  torch::Tensor Act(const torch::Tensor &obs) override {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(obs.to(model_locker_->device_));
    auto output = model.get_method("act")(input);
    auto action = output.toTensor();
    model_locker_->ReleaseModel(id);
    return action;
  }

private:
  std::shared_ptr<ModelLocker> model_locker_;
};
} // namespace rl::bridge
#endif // BRIDGE_RESEARCH_BRIDGE_ACTOR_H
