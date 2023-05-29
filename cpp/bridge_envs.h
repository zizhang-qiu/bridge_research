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

  [[nodiscard]] bool Terminated() const override;
  bool Reset() override;
  void Step(const TensorDict &reply) override;
  [[nodiscard]] TensorDict GetFeature() const override;
  [[nodiscard]] Player CurrentPlayer() const override;
  [[nodiscard]] std::vector<float> Returns() const override;
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] std::shared_ptr<BridgeBiddingState> GetState() const;
  static int GetFeatureSize() { return kAuctionTensorSize; }
  [[nodiscard]] int GetNumDeals() const { return num_deals_played_; }

 private:
  BridgeDeal current_deal_;
  std::shared_ptr<BridgeBiddingState> state_ = nullptr;
  std::shared_ptr<BridgeDealManager> deal_manager_;
  const std::vector<int> greedy_;
  int num_deals_played_ = -1;
};

class BridgeBiddingEnvWrapper {
 public:
  BridgeBiddingEnvWrapper(std::shared_ptr<BridgeDealManager> deal_manager,
                          const std::vector<int> &greedy,
                          std::shared_ptr<Replay> replay_buffer)
      : env_(std::move(deal_manager), greedy),
        replay_buffer_(std::move(replay_buffer)),
        transition_buffer_(bridge::kNumPlayers) {}

  bool Reset();
  [[nodiscard]] bool Terminated() const;
  void Step(const TensorDict &reply);
  [[nodiscard]] TensorDict GetFeature() {
    auto obs = env_.GetFeature();
    last_obs_ = tensor_dict::Clone(obs);
    return obs;
  }
  [[nodiscard]] std::string ToString() const {
    return env_.ToString();
  }
 private:
  BridgeBiddingEnv env_;
  MultiAgentTransitionBuffer transition_buffer_;
  std::shared_ptr<Replay> replay_buffer_ = nullptr;
  TensorDict last_obs_;
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
  [[nodiscard]] TensorDict GetFeature() const;
  [[nodiscard]] std::vector<std::shared_ptr<BridgeBiddingEnv>> GetEnvs() const {
    return envs_;
  }
  std::vector<int> GetReturns(Player player);

 private:
  std::vector<std::shared_ptr<BridgeBiddingEnv>> envs_;
};

class BridgeWrapperVecEnv{
 public:
  BridgeWrapperVecEnv() = default;
  void Push(const std::shared_ptr<BridgeBiddingEnvWrapper> &env) {
    envs_.emplace_back(env);
  }
  bool Reset();
  void Step(const TensorDict &reply);
  [[nodiscard]] bool AnyTerminated() const;
  [[nodiscard]] bool AllTerminated() const;
  [[nodiscard]] TensorDict GetFeature() const;
  [[nodiscard]] std::vector<std::shared_ptr<BridgeBiddingEnvWrapper>> GetEnvs() const {
    return envs_;
  }

 private:
  std::vector<std::shared_ptr<BridgeBiddingEnvWrapper>> envs_;
};

}  // namespace rl::bridge
#endif  // BRIDGE_RESEARCH_BRIDGE_ENVS_H_
