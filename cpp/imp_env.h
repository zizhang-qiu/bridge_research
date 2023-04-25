//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_IMP_ENV_H
#define BRIDGE_RESEARCH_IMP_ENV_H
#include "base.h"
#include "bridge_deal.h"
#include "bridge_envs.h"

namespace rl::bridge {
class ImpEnv : public Env {
 public:
  ImpEnv(std::shared_ptr<BridgeDealManager> deal_manager,
         std::vector<int> greedy,
         std::shared_ptr<ReplayBuffer> replay_buffer,
         bool eval)
      : deal_manager_(std::move(deal_manager)),
        greedy_(std::move(greedy)),
        replay_buffer(std::move(replay_buffer)),
        eval_(eval),
        transition_buffer_(kNumPlayers) {
  }

  TensorDict Reset() override {
    ++num_deals_played_;
    current_deal_ = deal_manager_->Next();
    auto state = std::make_shared<BridgeBiddingState>(current_deal_);
    states_[0] = state;
    current_state_ = 0;
    auto obs = MakeObsTensorDict(states_[0], greedy_[GetActingPlayer()]);
    last_obs_ = obs;
    return obs;
  }

  int GetNumDeals() const { return num_deals_played_; }

  ObsRewardTerminal Step(const TensorDict &reply) override {
    auto action = reply.at("a");
    auto action_int = action.item<int>();
    auto acting_player = GetActingPlayer();
    states_[current_state_]->ApplyAction(action_int);
    TensorDict obs;
    if (!eval_) {
      auto log_probs = reply.at("log_probs");
      transition_buffer_.PushObsActionLogProbs(acting_player,
                                               last_obs_.at("s"),
                                               action,
                                               log_probs
      );
    }
    if (current_state_ == 0 && states_[0]->Terminated()) {
      current_state_ = 1;
      states_[1] = std::make_shared<BridgeBiddingState>(current_deal_);
      obs = MakeObsTensorDict(states_[1], greedy_[GetActingPlayer()]);
    } else {
      obs = MakeObsTensorDict(states_[current_state_], greedy_[GetActingPlayer()]);
    }
    last_obs_ = obs;
    float r = 0.0f;
    bool t = Terminated();
    if (t && (!eval_)) {
//      std::cout << "reach here" << std::endl;
      std::vector<int> rewards = Returns();
      std::vector<double> normalized_reward(bridge::kNumPlayers);
      for (size_t i = 0; i < bridge::kNumPlayers; ++i) {
        normalized_reward[i] = static_cast<double>(rewards[i]) / bridge::kMaxImp;
      }
//      utils::PrintVector(normalized_reward);
      transition_buffer_.PushToReplayBuffer(replay_buffer, normalized_reward);
      transition_buffer_.Clear();
    }
    return std::make_tuple(obs, r, t);
  }

  Player GetActingPlayer() const {
    return (states_[current_state_]->CurrentPlayer() + current_state_) %
        kNumPlayers;
  }

  bool Terminated() const override {
    if (states_[0] == nullptr) {
      return true;
    }
    if (current_state_ == 0) {
      return false;
    }
    if (states_[1] == nullptr) {
      return false;
    }
    return states_[1]->Terminated();
  }

  std::vector<int> Returns() {
    RL_CHECK_TRUE(Terminated());
    double score_0 = states_[0]->Returns()[0];
    double score_1 = states_[1]->Returns()[0];
    int imp = GetImp(int(score_0), int(score_1));
    std::vector<int> ret = {imp, -imp, imp, -imp};
    return ret;
  }

  std::string ToString() {
    if (states_[0] == nullptr) {
      return "";
    }
    if (states_[1] == nullptr) {
      return states_[0]->ToString();
    }
    if (current_state_ == 0) {
      return states_[current_state_]->ToString();
    }
    return states_[0]->ToString() + "\n" + states_[1]->ToString();
  }

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

  void Push(const std::shared_ptr<ImpEnv> &env) {
    envs_.emplace_back(env);
  }

  int Size() const {
    return static_cast<int>(envs_.size());
  }

  TensorDict Reset(const TensorDict &obs) {
    std::vector<TensorDict> batch_obs;
    for (size_t i = 0; i < envs_.size(); ++i) {
      if (!envs_[i]->Terminated()) {
        batch_obs.emplace_back(tensor_dict::Index(obs, i));
      } else {
        auto env_obs = envs_[i]->Reset();
        batch_obs.emplace_back(env_obs);
      }
    }
    return tensor_dict::Stack(batch_obs, 0);
  }

  std::tuple<TensorDict, torch::Tensor, torch::Tensor> Step(const TensorDict &reply) {
    std::vector<TensorDict> obs_vector;
    torch::Tensor batch_reward = torch::zeros(envs_.size(), {torch::kFloat});
    torch::Tensor batch_terminal = torch::zeros(envs_.size(), {torch::kBool});
    TensorDict obs;
    float reward;
    bool terminal;
    for (size_t i = 0; i < envs_.size(); ++i) {
//      std::cout << i << std::endl;
      auto rep = tensor_dict::Index(reply, i);

      std::tie(obs, reward, terminal) = envs_[i]->Step(rep);

//      std::cout << "reach here" << std::endl;
      obs_vector.emplace_back(obs);
      batch_reward[i] = reward;
      batch_terminal[i] = terminal;
    }
    return std::make_tuple(tensor_dict::Stack(obs_vector, 0), batch_reward, batch_terminal);
  }

  bool AnyTerminated() const {
    for (size_t i = 0; i < envs_.size(); i++) {
      if (envs_[i]->Terminated()) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<std::shared_ptr<ImpEnv>> envs_;
};
}
#endif //BRIDGE_RESEARCH_IMP_ENV_H
