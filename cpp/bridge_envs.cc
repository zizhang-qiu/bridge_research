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
      {"s", torch::tensor(torch::ArrayRef<float>(observation))},
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

BridgeBiddingEnv::BridgeBiddingEnv(std::shared_ptr<BridgeDealManager> deal_manager,
                                   const std::vector<int> &greedy)
    : deal_manager_(std::move(deal_manager)),
      greedy_(greedy){
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

bool BridgeVecEnv::Reset() {
//  std::vector<TensorDict> batch_obs;
//  for (size_t i = 0; i < envs_.size(); ++i) {
//    if (!envs_[i]->Terminated()) {
//      batch_obs.emplace_back(tensor_dict::Index(obs, i));
//    } else {
//      auto env_obs = envs_[i]->Reset();
//      batch_obs.emplace_back(env_obs);
//    }
//  }
//  return tensor_dict::Stack(batch_obs, 0);
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (envs_[i]->Terminated()) {
      envs_[i]->Reset();
    }
  }
  return true;
}

void BridgeVecEnv::Step(const TensorDict &reply) {
  std::vector<TensorDict> obs_vector;
//  torch::Tensor batch_reward = torch::zeros(envs_.size(), {torch::kFloat});
//  torch::Tensor batch_terminal = torch::zeros(envs_.size(), {torch::kBool});
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (!envs_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      envs_[i]->Step(rep);
    }
//  return std::make_tuple(tensor_dict::Stack(obs_vector, 0), batch_reward, batch_terminal);
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
TensorDict BridgeVecEnv::GetFeatures() const {
  std::vector<TensorDict> obs_vec;
  for(size_t i=0; i<static_cast<int>(envs_.size()); ++i){
    if(!envs_[i]->Terminated()) {
      obs_vec.emplace_back(envs_[i]->GetFeature());
    }else{
      obs_vec.emplace_back(MakeTerminalObs(1));
    }
  }
  return tensor_dict::Stack(obs_vec, 0);
}
}