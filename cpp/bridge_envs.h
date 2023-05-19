//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_BRIDGE_ENVS_H_
#define BRIDGE_RESEARCH_BRIDGE_ENVS_H_
#include <utility>

#include "bridge_state.h"
#include "multi_agent_transition_buffer.h"
#include "rl/base.h"
#include "rl/tensor_dict.h"
#include "rl/types.h"
#include "torch/torch.h"

namespace rl::bridge {
torch::Tensor LegalActionsMask(const std::vector<Action> &legal_actions);

TensorDict MakeObsTensorDict(const std::shared_ptr<BridgeBiddingState> &state,
                             int greedy);

TensorDict MakeTerminalObs(int greedy);

class BridgeBiddingEnv : public Env {
 public:
  BridgeBiddingEnv(std::shared_ptr<BridgeDealManager> deal_manager,
                   const std::vector<int> &greedy);

  bool Terminated() const override;
  bool Reset() override;
  void Step(const TensorDict &reply) override;
  TensorDict GetFeature() const override;
  Player CurrentPlayer() const override;
  std::vector<float> Returns() const override;
  std::string ToString() const;
  std::shared_ptr<BridgeBiddingState> GetState() const;
  static int GetFeatureSize() { return kAuctionTensorSize; }
  int GetNumDeals() const { return num_deals_played_; }

 private:
  BridgeDeal current_deal_;
  std::shared_ptr<BridgeBiddingState> state_ = nullptr;
  std::shared_ptr<BridgeDealManager> deal_manager_;
  const std::vector<int> greedy_;
  int num_deals_played_ = -1;
};

class BridgeVecEnv {
 public:
  BridgeVecEnv() = default;

  void Push(const std::shared_ptr<BridgeBiddingEnv> &env) {
    envs_.emplace_back(env);
  }
  [[nodiscard]] int Size() const { return static_cast<int>(envs_.size()); }
  bool Reset();
  void Step(const TensorDict &reply);
  [[nodiscard]] bool AnyTerminated() const;
  [[nodiscard]] bool AllTerminated() const;
  [[nodiscard]] TensorDict GetFeatures() const;
  [[nodiscard]] std::vector<std::shared_ptr<BridgeBiddingEnv>> GetEnvs() const {
    return envs_;
  }
  std::vector<int> GetReturns(Player player);

 private:
  std::vector<std::shared_ptr<BridgeBiddingEnv>> envs_;
};
}  // namespace rl::bridge
#endif  // BRIDGE_RESEARCH_BRIDGE_ENVS_H_
