//
// Created by qzz on 2023/5/3.
//
#include "bridge_envs.h"
namespace rl::bridge {
torch::Tensor LegalActionsMask(const std::vector<Action> &legal_actions) {
  torch::Tensor mask = torch::zeros(kNumCalls, {torch::kFloat});
  for (auto action : legal_actions) {
    mask[action] = 1;
  }
  return mask;
}

TensorDict MakeObsTensorDict(const std::shared_ptr<BridgeBiddingState> &state,
                             int greedy) {
  auto observation = state->ObservationTensor();
  torch::Tensor legal_actions_mask;
  if (!state->Terminated()) {
    auto legal_actions = state->LegalActions();
    legal_actions_mask = LegalActionsMask(legal_actions);
  } else {
    legal_actions_mask = torch::ones(kNumCalls);
  }
  TensorDict obs = {
      {"s", torch::tensor(torch::ArrayRef<float>(observation))},
      {"legal_actions", legal_actions_mask},
      {"greedy", torch::tensor(greedy)}
  };

  return obs;
}

TensorDict MakeTerminalObs(int greedy) {
  TensorDict obs = {
      {"s", torch::zeros(bridge::kAuctionTensorSize)},
      {"legal_actions", torch::ones(bridge::kNumCalls)},
      {"greedy", torch::tensor(greedy)}
  };
  return obs;
}

BridgeBiddingEnv::BridgeBiddingEnv(std::shared_ptr<BridgeDealManager> deal_manager,
                                   const std::vector<int> &greedy,
                                   std::shared_ptr<ReplayBuffer> replay_buffer,
                                   bool use_par_score,
                                   bool eval)
    : deal_manager_(std::move(deal_manager)),
      greedy_(greedy),
      replay_buffer_(std::move(replay_buffer)),
      use_par_score_(use_par_score),
      eval_(eval),
      transition_buffer_(bridge::kNumPlayers) {
  RL_CHECK_EQ(greedy_.size(), bridge::kNumPlayers);
}

bool BridgeBiddingEnv::Terminated() const {
  if (state_ == nullptr) {
    return true;
  }
  return state_->Terminated();
}

TensorDict BridgeBiddingEnv::Reset() {
  ++num_deals_played_;
  current_deal_ = deal_manager_->Next();
  state_ = std::make_shared<BridgeBiddingState>(current_deal_);
  auto obs = MakeObsTensorDict(state_, greedy_[state_->CurrentPlayer()]);
  last_obs_ = obs;
  return obs;
}

ObsRewardTerminal BridgeBiddingEnv::Step(const TensorDict &reply) {
  float reward = 0.0f;
  auto current_player = state_->CurrentPlayer();
  auto action = reply.at("a");

  auto action_int = action.item<int>();
  TensorDict obs;

  state_->ApplyAction(action_int);
  obs = MakeObsTensorDict(state_, greedy_[state_->CurrentPlayer()]);

  bool terminal = state_->Terminated();
  if (!eval_) {
    auto log_probs = reply.at("log_probs");
    transition_buffer_.PushObsActionLogProbs(current_player,
                                             last_obs_.at("s"),
                                             action,
                                             log_probs);
    if (terminal) {
      std::vector<double> rewards = state_->Returns();
      if (use_par_score_) {

        auto par_score = static_cast<double>(current_deal_.par_score.value());
        for (size_t pl = 0; pl < bridge::kNumPlayers; pl++) {
          rewards[pl] = rewards[pl] - (pl % 2 == 0 ? par_score : -par_score);
        }
      } else {
        for (size_t pl = 0; pl < bridge::kNumPlayers; pl++) {
          rewards[pl] = rewards[pl] / kMaxScore;
        }
      }
      transition_buffer_.PushToReplayBuffer(replay_buffer_, rewards);
      transition_buffer_.Clear();
    }
  }
  last_obs_ = obs;
  return std::make_tuple(obs, reward, terminal);
}

Player BridgeBiddingEnv::GetCurrentPlayer() const {
  RL_CHECK_NOTNULL(state_);
  return state_->CurrentPlayer();
}

std::vector<double> BridgeBiddingEnv::Returns() const {
  RL_CHECK_NOTNULL(state_);
  return state_->Returns();
}

std::string BridgeBiddingEnv::ToString() const {
  if (state_ == nullptr) {
    return "Env not reset.";
  }
  return state_->ToString();
}

std::shared_ptr<BridgeBiddingState> BridgeBiddingEnv::GetState() const {
  RL_CHECK_NOTNULL(state_);
  return state_;
}

TensorDict BridgeVecEnv::Reset(const TensorDict &obs) {
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

std::tuple<TensorDict, torch::Tensor, torch::Tensor> BridgeVecEnv::Step(const TensorDict &reply) {
  std::vector<TensorDict> obs_vector;
  torch::Tensor batch_reward = torch::zeros(envs_.size(), {torch::kFloat});
  torch::Tensor batch_terminal = torch::zeros(envs_.size(), {torch::kBool});
  TensorDict obs;
  float reward;
  bool terminal;
  for (size_t i = 0; i < envs_.size(); ++i) {
    auto rep = tensor_dict::Index(reply, i);
    if (!envs_[i]->Terminated()) {
      std::tie(obs, reward, terminal) = envs_[i]->Step(rep);
    } else {
      obs = MakeTerminalObs(1);
      reward = 0.0f;
      terminal = true;
    }
    obs_vector.emplace_back(obs);
    batch_reward[i] = reward;
    batch_terminal[i] = terminal;
  }
  return std::make_tuple(tensor_dict::Stack(obs_vector, 0), batch_reward, batch_terminal);
}

bool BridgeVecEnv::AnyTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (envs_[i]->Terminated()) {
      return true;
    }
  }
  return false;
}

bool BridgeVecEnv::AllTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (!envs_[i]->Terminated()) {
      return false;
    }
  }
  return true;
}

std::vector<int> BridgeVecEnv::GetReturns(Player player) {
  int size = Size();
  std::vector<int> ret(size);
  for (size_t i = 0; i < size; ++i) {
    ret[i] = static_cast<int>(envs_[i]->Returns()[player]);
  }
  return ret;
}
}