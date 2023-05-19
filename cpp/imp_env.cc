//
// Created by qzz on 2023/5/3.
//
#include "imp_env.h"

namespace rl::bridge {

ImpEnv::ImpEnv(std::shared_ptr<BridgeDealManager> deal_manager,
               const std::vector<int> &greedy)
    : deal_manager_(std::move(deal_manager)),
      greedy_(greedy) {}

bool ImpEnv::Reset() {
  ++num_deals_played_;
  current_deal_ = deal_manager_->Next();
  auto state0 = std::make_shared<BridgeBiddingState>(current_deal_);
  states_[0] = state0;
  auto state1 = std::make_shared<BridgeBiddingState>(current_deal_);
  states_[1] = state1;
  current_state_ = 0;
//  auto obs = MakeObsTensorDict(states_[0], greedy_[GetActingPlayer()]);
  return true;
}

void ImpEnv::Step(const TensorDict &reply) {
  auto action = reply.at("a");
  auto action_int = action.item<int>();
  states_[current_state_]->ApplyAction(action_int);
  TensorDict obs;
  if (current_state_ == 0 && states_[0]->Terminated()) {
    current_state_ = 1;
  }
//  obs = MakeObsTensorDict(states_[current_state_], greedy_[GetActingPlayer()]);
//  return std::make_tuple(obs, r, t);
}

bool ImpEnv::Terminated() const {
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

std::vector<float> ImpEnv::Returns() const {
  RL_CHECK_TRUE(Terminated())
  double score_0 = states_[0]->Returns()[0];
  double score_1 = states_[1]->Returns()[0];
  auto imp = static_cast<float>(GetImp(int(score_0), int(score_1)));
  std::vector<float> ret = {imp, -imp, imp, -imp};
  return ret;
}

std::string ImpEnv::ToString() const {
  if (states_[0] == nullptr) {
    return "Env not reset.";
  }
  if (current_state_ == 0) {
    return states_[current_state_]->ToString();
  }
  return states_[0]->ToString() + "\n" + states_[1]->ToString();
}

Player ImpEnv::CurrentPlayer() const {
  RL_CHECK_NOTNULL(states_[0]);
  return states_[current_state_]->CurrentPlayer();
}

TensorDict ImpEnv::GetFeature() const {
  auto obs = MakeObsTensorDict(states_[current_state_], greedy_[GetActingPlayer()]);
  return obs;
}

bool ImpVecEnv::Reset() {
//  std::vector<TensorDict> batch_obs;
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (envs_[i]->Terminated()) {
//      batch_obs.emplace_back(tensor_dict::Index(obs, i));
      envs_[i]->Reset();
//      batch_obs.emplace_back(env_obs);
    }
  }
//  return tensor_dict::Stack(batch_obs, 0);
  return true;
}

void ImpVecEnv::Step(const TensorDict &reply) {
//  std::vector<TensorDict> obs_vector;
//  torch::Tensor batch_reward = torch::zeros(envs_.size(), {torch::kFloat});
//  torch::Tensor batch_terminal = torch::zeros(envs_.size(), {torch::kBool});
//  TensorDict obs;
//  float reward;
//  bool terminal;
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (!envs_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      envs_[i]->Step(rep);
    }
//    obs_vector.emplace_back(obs);
//    batch_reward[i] = reward;
//    batch_terminal[i] = terminal;
  }
//  return std::make_tuple(tensor_dict::Stack(obs_vector, 0), batch_reward, batch_terminal);
}

bool ImpVecEnv::AnyTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (envs_[i]->Terminated()) {
      return true;
    }
  }
  return false;
}

bool ImpVecEnv::AllTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (!envs_[i]->Terminated()) {
      return false;
    }
  }
  return true;
}

TensorDict ImpVecEnv::GetFeature() const {
  std::vector<TensorDict> obs_vec;
  for (size_t i = 0; i < envs_.size(); i++) {
    obs_vec.emplace_back(envs_[i]->GetFeature());
  }
  return tensor_dict::Stack(obs_vec, 0);
}

} // namespace rl::bridge