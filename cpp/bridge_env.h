//
// Created by qzz on 2023/2/27.
//
#include "base.h"
#include "bridge_state.h"
#include "logging.h"
#include "pybind11/numpy.h"
#include "types.h"
#include <iostream>
#include <random>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <utility>
#include <vector>

#ifndef BRIDGE_RESEARCH_BRIDGE_ENV_H
#define BRIDGE_RESEARCH_BRIDGE_ENV_H
namespace rl::bridge {
inline constexpr int kObsTensorSize = kAuctionTensorSize + kNumCalls + 1;

torch::Tensor LegalActionsMask(const std::vector<Action> &legal_actions) {
  torch::Tensor mask = torch::zeros(kNumCalls, {torch::kFloat});
  for (auto action : legal_actions) {
    mask[action] = 1;
  }
  return mask;
}

torch::Tensor MakeObsTensor(const std::shared_ptr<BridgeBiddingState> &state,
                            int greedy) {
  torch::Tensor obs = torch::zeros(kObsTensorSize, {torch::kFloat});
  auto observation = state->ObservationTensor();
  obs.slice(0, 0, kAuctionTensorSize)
      .copy_(torch::tensor(torch::ArrayRef<float>(observation)));
  auto legal_actions = state->LegalActions();
  auto legal_actions_mask = LegalActionsMask(legal_actions);
  obs.slice(0, kAuctionTensorSize, kAuctionTensorSize + kNumCalls)
      .copy_(legal_actions_mask);
  obs[-1] = greedy;

  return obs;
}

class BridgeBiddingEnv : public Env {
public:
  BridgeBiddingEnv(const std::vector<std::vector<Action>> &card_trajectories,
                   const std::vector<std::vector<int>> &ddts,
                   const std::vector<int> &greedy)
      : card_trajectories_(card_trajectories), ddts_(ddts), greedy_(greedy) {
    RL_CHECK_GT(card_trajectories_.size(), 0);
    RL_CHECK_EQ(card_trajectories_.size(), ddts_.size());
    RL_CHECK_EQ(greedy.size(), kNumPlayers);
    size_ = ddts.size();
    current_cards_ = card_trajectories_[cursor_];
    current_ddt_ = ddts_[cursor_];
  }

  BridgeBiddingEnv(const std::vector<Action> &cards,
                   const std::vector<int> &ddt,
                   const std::vector<int> &greedy) {
    RL_CHECK_EQ(cards.size(), kNumCards);
    RL_CHECK_EQ(ddt.size(), kDoubleDummyResultSize);
    std::vector<std::vector<Action>> card_trajectories = {cards};
    std::vector<std::vector<int>> ddts = {ddt};
    card_trajectories_ = card_trajectories;
    ddts_ = ddts;
    greedy_ = greedy;
    size_ = 1;
    current_cards_ = card_trajectories_[cursor_];
    current_ddt_ = ddts_[cursor_];
  }

  torch::Tensor Reset() override {
    num_states_ += 1;
    current_cards_ = card_trajectories_[cursor_];
    current_ddt_ = ddts_[cursor_];
    cursor_ = (cursor_ + 1) % size_;
    BridgeDeal deal{current_cards_, dealer_, false, false, current_ddt_};

    state_ = std::make_shared<BridgeBiddingState>(deal);

    auto obs = MakeObsTensor(state_, greedy_[state_->CurrentPlayer()]);
    return obs;
  }

  std::tuple<torch::Tensor, float, bool>
  Step(const torch::Tensor &action) override {
    auto action_int = action.item<int>();
    state_->ApplyAction(action_int);

    auto obs = MakeObsTensor(state_, greedy_[state_->CurrentPlayer()]);
    float reward = 0.0f;
    bool terminated = state_->Terminated();
    return std::make_tuple(obs, reward, terminated);
  }

  std::string ToString() const {
    if (state_ != nullptr) {
      return state_->ToString();
    }
    return "";
  }

  bool Terminated() const override {
    if (state_ == nullptr) {
      return true;
    }
    return state_->Terminated();
  }

  std::vector<double> Returns() {
    RL_CHECK_TRUE(state_ != nullptr);
    return state_->Returns();
  }

  Player CurrentPlayer() {
    RL_CHECK_TRUE(state_ != nullptr);
    return state_->CurrentPlayer();
  }

  std::tuple<std::vector<Action>, std::vector<int>>
  GetCurrentCardsAndDDT() const {
    return std::make_tuple(current_cards_, current_ddt_);
  }

  std::shared_ptr<BridgeBiddingEnv> MakeSubEnv() {
    std::vector<std::vector<Action>> cards_trajectories = {current_cards_};

    std::vector<std::vector<int>> ddts = {current_ddt_};
    return std::make_shared<BridgeBiddingEnv>(cards_trajectories, ddts,
                                              greedy_);
  }

  std::shared_ptr<BridgeBiddingState> GetState() {
    RL_CHECK_TRUE(state_ != nullptr);
    return state_;
  }

  int NumStates() const { return num_states_; }

private:
  std::shared_ptr<BridgeBiddingState> state_ = nullptr;
  std::vector<std::vector<Action>> card_trajectories_;
  std::vector<std::vector<int>> ddts_;
  std::vector<Action> current_cards_;
  std::vector<int> current_ddt_;
  int size_;
  int cursor_ = 0;
  int dealer_ = Seat::kNorth;
  std::vector<int> greedy_;
  int num_states_ = 0;
};

// This env use real score - par score as reward.
class BridgeBiddingEnv2 : public Env {
public:
  BridgeBiddingEnv2(const std::vector<std::vector<Action>> &card_trajectories,
                    const std::vector<std::vector<int>> &ddts,
                    const std::vector<int>& par_scores,
                    const std::vector<int> &greedy)
      : card_trajectories_(card_trajectories), ddts_(ddts),
        par_scores_(par_scores), greedy_(greedy) {
    RL_CHECK_GT(card_trajectories_.size(), 0);
    RL_CHECK_EQ(card_trajectories_.size(), ddts_.size());
    RL_CHECK_EQ(greedy.size(), kNumPlayers);
    size_ = ddts.size();
    current_cards_ = card_trajectories_[cursor_];
    current_ddt_ = ddts_[cursor_];
    current_par_score_ = (double)par_scores_[cursor_];
  }

  torch::Tensor Reset() override {
    num_states_ += 1;
    current_cards_ = card_trajectories_[cursor_];
    current_ddt_ = ddts_[cursor_];
    current_par_score_ = (double)(par_scores_[cursor_]);
    cursor_ = (cursor_ + 1) % size_;

    BridgeDeal deal{current_cards_, dealer_, false, false, current_ddt_};

    state_ = std::make_shared<BridgeBiddingState>(deal);

    auto obs = MakeObsTensor(state_, greedy_[state_->CurrentPlayer()]);
    return obs;
  }

  std::tuple<torch::Tensor, float, bool>
  Step(const torch::Tensor &action) override {
    auto action_int = action.item<int>();
    state_->ApplyAction(action_int);

    auto obs = MakeObsTensor(state_, greedy_[state_->CurrentPlayer()]);
    float reward = 0.0f;
    bool terminated = state_->Terminated();
    return std::make_tuple(obs, reward, terminated);
  }

  std::string ToString() const {
    if (state_ != nullptr) {
      return state_->ToString();
    }
    return "";
  }

  bool Terminated() const override {
    if (state_ == nullptr) {
      return true;
    }
    return state_->Terminated();
  }

  std::vector<double> Returns() {
    RL_CHECK_TRUE(state_ != nullptr);
    auto raw_scores = state_->Returns();
    std::vector<double> ret(kNumPlayers);
    for (size_t i = 0; i < kNumPlayers; ++i) {
      ret[i] = raw_scores[i] -
               (i % 2 == 0 ? current_par_score_ : -current_par_score_);
    }
    return ret;
  }

  Player CurrentPlayer() {
    RL_CHECK_TRUE(state_ != nullptr);
    return state_->CurrentPlayer();
  }

  std::tuple<std::vector<Action>, std::vector<int>>
  GetCurrentCardsAndDDT() const {
    return std::make_tuple(current_cards_, current_ddt_);
  }


  std::shared_ptr<BridgeBiddingState> GetState() {
    RL_CHECK_TRUE(state_ != nullptr);
    return state_;
  }

  int NumStates() const { return num_states_; }

private:
  std::shared_ptr<BridgeBiddingState> state_ = nullptr;
  std::vector<std::vector<Action>> card_trajectories_;
  std::vector<std::vector<int>> ddts_;
  std::vector<int> par_scores_;
  double current_par_score_;
  std::vector<Action> current_cards_;
  std::vector<int> current_ddt_;
  int size_;
  int cursor_ = 0;
  int dealer_ = Seat::kNorth;
  std::vector<int> greedy_;
  int num_states_ = 0;
};

// a vectorized env for faster evaluation
class BridgeVecEnv {
public:
  BridgeVecEnv() = default;

  virtual ~BridgeVecEnv() = default;

  void Append(const std::shared_ptr<BridgeBiddingEnv> &env) {
    envs_.push_back(env);
  }

  int Size() { return envs_.size(); }

  // we only call reset at start or after every env is terminated
  torch::Tensor Reset() {
    auto batch_obs = torch::zeros({Size(), kObsTensorSize}, {torch::kFloat});
    for (int i = 0; i < envs_.size(); i++) {
      torch::Tensor obs = envs_[i]->Reset();
      batch_obs[i] = obs;
    }
    return batch_obs;
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  Step(const torch::Tensor &action) {
    auto batch_obs = torch::zeros({Size(), kObsTensorSize});
    auto batch_reward = torch::zeros(Size(), {torch::kFloat});
    auto batch_terminal = torch::zeros(Size(), {torch::kBool});
    torch::Tensor obs;
    float r;
    bool t;
    for (int i = 0; i < envs_.size(); i++) {
      if (envs_[i]->Terminated()) {
        batch_terminal[i] = true;
      } else {
        std::tie(obs, r, t) = envs_[i]->Step(action[i]);
        batch_obs[i] = obs;
        batch_reward[i] = r;
      }
    }
    return std::make_tuple(batch_obs, batch_reward, batch_terminal);
  }

  bool AllTerminated() const {
    for (int i = 0; i < envs_.size(); i++) {
      if (!envs_[i]->Terminated()) {
        return false;
      }
    }
    return true;
  }

  void Display(int num_envs) {
    RL_CHECK_LE(num_envs, Size());
    for (int i = 0; i < num_envs; i++) {
      std::cout << "env " << i << std::endl;
      std::cout << envs_[i]->ToString() << std::endl;
    }
  }

  std::vector<double> Returns(Player player) {
    RL_CHECK_TRUE(AllTerminated());
    std::vector<double> ret(Size());
    for (int i = 0; i < envs_.size(); i++) {
      ret[i] = envs_[i]->Returns()[player];
    }
    return ret;
  }

private:
  std::vector<std::shared_ptr<BridgeBiddingEnv>> envs_;
};

class ImpEnv : public Env {
public:
  ImpEnv(const std::vector<std::vector<Action>> &cards,
         const std::vector<std::vector<int>> &ddts, std::vector<int> greedy,
         bool save_history_imps)
      : cards_(cards), ddts_(ddts), greedy_(std::move(greedy)),
        save_history_imps_(save_history_imps) {
    RL_CHECK_GT(cards_.size(), 0);
    RL_CHECK_EQ(cards_.size(), ddts.size());
    RL_CHECK_EQ(greedy_.size(), kNumPlayers);
    size_ = cards_.size();
  }

  torch::Tensor Reset() override {
    current_cards_ = cards_[cursor_];
    current_ddt_ = ddts_[cursor_];
    cursor_ = (cursor_ + 1) % size_;
    num_states_++;
    BridgeDeal deal{current_cards_, kNorth, false, false, current_ddt_};
    auto state = std::make_shared<BridgeBiddingState>(deal);
    states_[0] = state;
    current_state_ = 0;
    auto obs_tensor = MakeObsTensor(states_[0], greedy_[ActingPlayer()]);
    return obs_tensor;
  }

  int NumStates() const { return num_states_; }

  std::tuple<torch::Tensor, float, bool>
  Step(const torch::Tensor &action) override {
    int action_int = action.item<int>();
    states_[current_state_]->ApplyAction(action_int);
    torch::Tensor obs_tensor;
    if (current_state_ == 0 && states_[0]->Terminated()) {
      current_state_ = 1;
      BridgeDeal deal{current_cards_, kNorth, false, false, current_ddt_};
      states_[1] = std::make_shared<BridgeBiddingState>(deal);
      obs_tensor = MakeObsTensor(states_[1], greedy_[ActingPlayer()]);
    } else {
      obs_tensor =
          MakeObsTensor(states_[current_state_], greedy_[ActingPlayer()]);
    }
    float r = 0.0f;
    bool t = Terminated();
    if (t && save_history_imps_) {
      auto imp = Returns()[0];
      history_imps_.emplace_back(imp);
    }
    return std::make_tuple(obs_tensor, r, t);
  }

  Player ActingPlayer() const {
    return (states_[current_state_]->CurrentPlayer() + current_state_) %
           kNumPlayers;
  }

  bool Terminated() const override {
    if (states_[0] == nullptr) {
      return false;
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
    std::vector<int> ret = {imp, -imp};
    return ret;
  }

  std::vector<int> HistoryImps() const { return history_imps_; }

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
  const std::vector<std::vector<Action>> cards_;
  const std::vector<std::vector<int>> ddts_;
  std::vector<int> current_cards_;
  std::vector<int> current_ddt_;
  std::vector<std::shared_ptr<BridgeBiddingState>> states_ = {nullptr, nullptr};
  int current_state_ = 0;
  const std::vector<int> greedy_;
  int cursor_ = 0;
  int size_;
  int num_states_ = 0;
  bool save_history_imps_;
  std::vector<int> history_imps_;
};
} // namespace rl::bridge

#endif // BRIDGE_RESEARCH_BRIDGE_ENV_H
