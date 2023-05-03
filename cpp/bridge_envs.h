//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_BRIDGE_ENVS_H_
#define BRIDGE_RESEARCH_BRIDGE_ENVS_H_
#include <utility>

#include "rl/base.h"
#include "bridge_state.h"
#include "rl/tensor_dict.h"
#include "torch/torch.h"
#include "rl/types.h"
#include "multi_agent_transition_buffer.h"
namespace rl::bridge {
torch::Tensor LegalActionsMask(const std::vector<Action> &legal_actions);

TensorDict MakeObsTensorDict(const std::shared_ptr<BridgeBiddingState> &state,
                             int greedy);

TensorDict MakeTerminalObs(int greedy);

class BridgeBiddingEnv : public Env {
 public:
  BridgeBiddingEnv(std::shared_ptr<BridgeDealManager> deal_manager,
                   const std::vector<int> &greedy,
                   std::shared_ptr<ReplayBuffer> replay_buffer,
                   bool use_par_score,
                   bool eval);

  bool Terminated() const override;
  TensorDict Reset() override;
  ObsRewardTerminal Step(const TensorDict &reply) override;
  Player GetCurrentPlayer() const;
  std::vector<double> Returns() const;
  std::string ToString() const;
  std::shared_ptr<BridgeBiddingState> GetState() const;
  static int GetFeatureSize() {return kAuctionTensorSize;}
  int GetNumDeals() const {return num_deals_played_;}

 private:
  BridgeDeal current_deal_;
  std::shared_ptr<BridgeBiddingState> state_ = nullptr;
  std::shared_ptr<BridgeDealManager> deal_manager_;
  const std::vector<int> greedy_;
  int num_deals_played_ = -1;
  TensorDict last_obs_;
  const bool eval_;
  // using raw score - par score as reward
  const bool use_par_score_;
  MultiAgentTransitionBuffer transition_buffer_;
  std::shared_ptr<ReplayBuffer> replay_buffer_ = nullptr;
};

class BridgeVecEnv {
 public:
  BridgeVecEnv() = default;

  void Push(const std::shared_ptr<BridgeBiddingEnv> &env) {envs_.emplace_back(env);}
  int Size() const {return static_cast<int>(envs_.size());}
  TensorDict Reset(const TensorDict &obs);
  std::tuple<TensorDict, torch::Tensor, torch::Tensor> Step(const TensorDict &reply);
  bool AnyTerminated() const;
  bool AllTerminated() const;
  std::vector<std::shared_ptr<BridgeBiddingEnv>> GetEnvs() const {return envs_;}
  std::vector<int> GetReturns(Player player);

 private:
  std::vector<std::shared_ptr<BridgeBiddingEnv>> envs_;
};
}
#endif // BRIDGE_RESEARCH_BRIDGE_ENVS_H_
