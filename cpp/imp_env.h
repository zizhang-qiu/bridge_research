//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_IMP_ENV_H
#define BRIDGE_RESEARCH_IMP_ENV_H
#include <utility>

#include "rl/base.h"
#include "bridge_deal.h"
#include "bridge_envs.h"

namespace rl::bridge {

class ImpEnv : public Env {
 public:
  ImpEnv(std::shared_ptr<BridgeDealManager> deal_manager,
         const std::vector<int> &greedy);

  bool Reset() override;
  [[nodiscard]] int GetNumDeals() const { return num_deals_played_; }
  void Step(const TensorDict &reply) override;
  [[nodiscard]] Player GetActingPlayer() const {
    return (states_[current_state_]->CurrentPlayer() + current_state_) % kNumPlayers;
  }
  [[nodiscard]] Player CurrentPlayer() const override;
  [[nodiscard]] bool Terminated() const override;
  [[nodiscard]] std::vector<float> Returns() const override;
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] TensorDict GetFeature() const override;

 private:
  BridgeDeal current_deal_;
  std::shared_ptr<BridgeDealManager> deal_manager_;
  std::vector<std::shared_ptr<BridgeBiddingState>> states_ = {nullptr, nullptr};
  int current_state_ = 0;
  const std::vector<int> greedy_;
  int num_deals_played_ = -1;
};

class ImpEnvWrapper {
 public:
  ImpEnvWrapper(std::shared_ptr<BridgeDealManager> deal_manager,
                const std::vector<int> &greedy,
                std::shared_ptr<Replay> replay_buffer)
      : env_(std::move(deal_manager), greedy),
        replay_buffer_(std::move(replay_buffer)),
        transition_buffer_(bridge::kNumPlayers){}

  bool Reset() {
    env_.Reset();
    return true;
  }

  void Step(const TensorDict &reply);

  [[nodiscard]] TensorDict GetFeature() {
    auto obs = env_.GetFeature();
    last_obs_ = tensor_dict::Clone(obs);
    return obs;
  }

  [[nodiscard]] bool Terminated() const {
    return env_.Terminated();
  }

  [[nodiscard]] std::string ToString() const {
    return env_.ToString();
  }

 private:
  ImpEnv env_;
  MultiAgentTransitionBuffer transition_buffer_;
  std::shared_ptr<Replay> replay_buffer_;
  TensorDict last_obs_;
};

class ImpVecEnv {
 public:
  ImpVecEnv() = default;

  void Push(const std::shared_ptr<ImpEnvWrapper> &env) { envs_.emplace_back(env); }

  [[nodiscard]] int Size() const { return static_cast<int>(envs_.size()); }

  bool Reset();

  void Step(const TensorDict &reply);

  [[nodiscard]] bool AnyTerminated() const;

  [[nodiscard]] bool AllTerminated() const;

  [[nodiscard]] TensorDict GetFeature() const;

  [[nodiscard]] std::vector<std::shared_ptr<ImpEnvWrapper>> GetEnvs() const {
    return envs_;
  }

 private:
  std::vector<std::shared_ptr<ImpEnvWrapper>> envs_;
};
}
#endif //BRIDGE_RESEARCH_IMP_ENV_H
