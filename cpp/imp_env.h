//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_IMP_ENV_H
#define BRIDGE_RESEARCH_IMP_ENV_H
#include "rl/base.h"
#include "bridge_deal.h"
#include "bridge_envs.h"

namespace rl::bridge {
class ImpEnv : public Env {
 public:
  ImpEnv(std::shared_ptr<BridgeDealManager> deal_manager,
         std::vector<int> greedy,
         std::shared_ptr<ReplayBuffer> replay_buffer,
         bool eval);

  TensorDict Reset() override;
  int GetNumDeals() const { return num_deals_played_; }
  ObsRewardTerminal Step(const TensorDict &reply) override;
  Player GetActingPlayer() const { return (states_[current_state_]->CurrentPlayer() + current_state_) % kNumPlayers; }
  bool Terminated() const override;
  std::vector<int> Returns();
  std::string ToString();

 private:
  BridgeDeal current_deal_;
  std::shared_ptr<BridgeDealManager> deal_manager_;
  std::vector<std::shared_ptr<BridgeBiddingState>> states_ = {nullptr, nullptr};
  int current_state_ = 0;
  const std::vector<int> greedy_;
  int num_deals_played_ = -1;
  MultiAgentTransitionBuffer transition_buffer_;
  TensorDict last_obs_;
  bool eval_;
  std::shared_ptr<ReplayBuffer> replay_buffer = nullptr;
};

class ImpVecEnv {
 public:
  ImpVecEnv() = default;

  void Push(const std::shared_ptr<ImpEnv> &env) { envs_.emplace_back(env); }

  int Size() const { return static_cast<int>(envs_.size()); }

  TensorDict Reset(const TensorDict &obs);

  std::tuple<TensorDict, torch::Tensor, torch::Tensor> Step(const TensorDict &reply);

  bool AnyTerminated() const;

 private:
  std::vector<std::shared_ptr<ImpEnv>> envs_;
};
}
#endif //BRIDGE_RESEARCH_IMP_ENV_H
