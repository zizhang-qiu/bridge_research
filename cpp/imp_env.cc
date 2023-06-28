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
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (envs_[i]->Terminated()) {
      envs_[i]->Reset();
    }
  }
  return true;
}

void ImpVecEnv::Step(const TensorDict &reply) {
  for (size_t i = 0; i < envs_.size(); ++i) {
    if (!envs_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      envs_[i]->Step(rep);
    }
  }
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

void ImpEnvWrapper::Step(const TensorDict &reply) {
  auto acting_player = env_.GetActingPlayer();
  transition_buffer_.PushObsAndReply(acting_player, last_obs_, reply);
//  if(add_random_illegal_transitions_ && torch::rand(1).item<float>() > 0.5){
//    auto illegal_reply = SampleIllegalAction(last_obs_, reply);
//  }
  env_.Step(reply);
  if (env_.Terminated()) {
    std::vector<float> rewards = env_.Returns();
    std::vector<float> normalized_rewards(kNumPlayers);
    for (int i = 0; i < kNumPlayers; ++i) {
      normalized_rewards[i] = rewards[i] / static_cast<float>(bridge::kMaxImp);
    }

    transition_buffer_.PushToReplayBuffer(replay_buffer_, normalized_rewards);
    transition_buffer_.Clear();
  }
}
} // namespace rl::bridge