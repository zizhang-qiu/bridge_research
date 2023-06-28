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
  auto hidden_info = state->HiddenObservationTensor();
  auto perfect_observation = utils::ConcatenateVectors(observation, hidden_info);
  torch::Tensor legal_actions_mask;
  if (!state->Terminated()) {
    auto legal_actions = state->LegalActions();
    legal_actions_mask = LegalActionsMask(legal_actions);
  } else {
    legal_actions_mask = torch::ones(kNumCalls);
  }
  TensorDict obs = {
      {"perfect_s", torch::tensor(torch::ArrayRef<float>(perfect_observation))},
      {"s", {torch::tensor(torch::ArrayRef<float>(observation))}},
      {"legal_actions", legal_actions_mask},
      {"greedy", torch::tensor(greedy)}
  };

  return obs;
}

TensorDict MakeTerminalObs(int greedy) {
  TensorDict obs = {
      {"perfect_s", torch::zeros(bridge::kPerfectInfoTensorSize)},
      {"s", torch::zeros(bridge::kAuctionTensorSize)},
      {"legal_actions", torch::ones(bridge::kNumCalls)},
      {"greedy", torch::tensor(greedy)}
  };
  return obs;
}

BridgeBiddingEnv::BridgeBiddingEnv(std::shared_ptr<DealManager> deal_manager,
                                   const std::vector<int> &greedy)
    : deal_manager_(std::move(deal_manager)),
      greedy_(greedy) {
  RL_CHECK_EQ(greedy_.size(), bridge::kNumPlayers);
}

bool BridgeBiddingEnv::Terminated() const {
  if (state_ == nullptr) {
    return true;
  }
  return state_->Terminated();
}

bool BridgeBiddingEnv::Reset() {
  ++num_deals_played_;
  current_deal_ = deal_manager_->Next();
  state_ = std::make_shared<BridgeBiddingState>(current_deal_);
//  auto obs = MakeObsTensorDict(state_, greedy_[state_->CurrentPlayer()]);
  return true;
}

void BridgeBiddingEnv::Step(const TensorDict &reply) {
  auto action = reply.at("a");

  auto action_int = action.item<int>();
  TensorDict obs;

  state_->ApplyAction(action_int);
//  obs = MakeObsTensorDict(state_, greedy_[state_->CurrentPlayer()]);
//  return std::make_tuple(obs, reward, terminal);
}

Player BridgeBiddingEnv::CurrentPlayer() const {
  RL_CHECK_NOTNULL(state_);
  return state_->CurrentPlayer();
}

std::vector<float> BridgeBiddingEnv::Returns() const {
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

TensorDict BridgeBiddingEnv::GetFeature() const {
  RL_CHECK_NOTNULL(state_);
  auto current_player = state_->CurrentPlayer();
  return MakeObsTensorDict(state_, greedy_[current_player]);
}
TensorDict BridgeBiddingEnv::GetBeliefFeature() const {
  RL_CHECK_NOTNULL(state_);
  torch::Tensor belief = torch::tensor(state_->HiddenObservationTensor(), torch::kFloat32);
  return {
      {"belief", belief}
  };
}

bool BridgeBiddingEnvWrapper::Reset() {
  env_.Reset();
  return true;
}

bool BridgeBiddingEnvWrapper::Terminated() const {
  return env_.Terminated();
}

void BridgeBiddingEnvWrapper::Step(const TensorDict &reply) {
  auto acting_player = env_.CurrentPlayer();
  if (replay_buffer_ != nullptr) {
    transition_buffer_.PushObsAndReply(acting_player, last_obs_, reply);
  }
  env_.Step(reply);
  if (env_.Terminated() && replay_buffer_ != nullptr) {
    std::vector<float> rewards = env_.Returns();
    std::vector<float> normalized_rewards(kNumPlayers);
    for (int i = 0; i < kNumPlayers; ++i) {
      normalized_rewards[i] = rewards[i] / static_cast<float>(bridge::kMaxScore);
    }
    transition_buffer_.PushToReplayBuffer(replay_buffer_, normalized_rewards);
    transition_buffer_.Clear();
  }
}

bool BridgeVecEnv::Reset() {
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (envs_[i]->Terminated()) {
      envs_[i]->Reset();
    }
  }
  return true;
}

void BridgeVecEnv::Step(const TensorDict &reply) {
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (!envs_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      envs_[i]->Step(rep);
    }
  }
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

TensorDict BridgeVecEnv::GetFeature() const {
  std::vector<TensorDict> obs_vec;
  for (size_t i = 0; i < static_cast<int>(envs_.size()); ++i) {
    if (!envs_[i]->Terminated()) {
      obs_vec.emplace_back(envs_[i]->GetFeature());
    } else {
      obs_vec.emplace_back(MakeTerminalObs(1));
    }
  }
  return tensor_dict::Stack(obs_vec, 0);
}

std::vector<std::vector<int>> BridgeVecEnv::GetHistories() const {
  std::vector<std::vector<int>> histories(envs_.size());
  for (size_t i = 0; i < envs_.size(); ++i) {
    histories[i] = envs_[i]->GetState()->History();
  }
  return histories;
}

TensorDict BridgeVecEnv::GetBeliefFeature() const {
  std::vector<TensorDict> belief_vec;
  for (size_t i = 0; i < static_cast<int>(envs_.size()); ++i){
    belief_vec.push_back(envs_[i]->GetBeliefFeature());
  }
  return tensor_dict::Stack(belief_vec, 0);
}

bool BridgeWrapperVecEnv::Reset() {
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (envs_[i]->Terminated()) {
      envs_[i]->Reset();
    }
  }
  return true;
}

void BridgeWrapperVecEnv::Step(const TensorDict &reply) {
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (!envs_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      envs_[i]->Step(rep);
    }
  }
}

bool BridgeWrapperVecEnv::AnyTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (envs_[i]->Terminated()) {
      return true;
    }
  }
  return false;
}

bool BridgeWrapperVecEnv::AllTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (!envs_[i]->Terminated()) {
      return false;
    }
  }
  return true;
}

TensorDict BridgeWrapperVecEnv::GetFeature() const {
  std::vector<TensorDict> obs_vec;
  for (size_t i = 0; i < static_cast<int>(envs_.size()); ++i) {
    if (!envs_[i]->Terminated()) {
      obs_vec.emplace_back(envs_[i]->GetFeature());
    } else {
      obs_vec.emplace_back(MakeTerminalObs(1));
    }
  }
  return tensor_dict::Stack(obs_vec, 0);
}

}