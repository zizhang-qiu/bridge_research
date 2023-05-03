//
// Created by qzz on 2023/5/3.
//
#include "imp_env.h"

namespace rl::bridge {

ImpEnv::ImpEnv(std::shared_ptr<BridgeDealManager> deal_manager,
               std::vector<int> greedy,
               std::shared_ptr<ReplayBuffer> replay_buffer,
               bool eval)
    : deal_manager_(std::move(deal_manager)),
      greedy_(std::move(greedy)),
      replay_buffer(std::move(replay_buffer)),
      eval_(eval),
      transition_buffer_(kNumPlayers) {}

TensorDict ImpEnv::Reset() {
  ++num_deals_played_;
  current_deal_ = deal_manager_->Next();
  auto state0 = std::make_shared<BridgeBiddingState>(current_deal_);
  states_[0] = state0;
  auto state1 = std::make_shared<BridgeBiddingState>(current_deal_);
  states_[1] = state1;
  current_state_ = 0;
  auto obs = MakeObsTensorDict(states_[0], greedy_[GetActingPlayer()]);
  last_obs_ = obs;
  return obs;
}

ObsRewardTerminal ImpEnv::Step(const TensorDict &reply) {
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
  }
  obs = MakeObsTensorDict(states_[current_state_], greedy_[GetActingPlayer()]);

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

std::vector<int> ImpEnv::Returns() {
  RL_CHECK_TRUE(Terminated());
  double score_0 = states_[0]->Returns()[0];
  double score_1 = states_[1]->Returns()[0];
  int imp = GetImp(int(score_0), int(score_1));
  std::vector<int> ret = {imp, -imp, imp, -imp};
  return ret;
}

std::string ImpEnv::ToString() {
  if (states_[0] == nullptr) {
    return "Env not reset.";
  }
  if (current_state_ == 0) {
    return states_[current_state_]->ToString();
  }
  return states_[0]->ToString() + "\n" + states_[1]->ToString();
}

TensorDict ImpVecEnv::Reset(const TensorDict &obs) {
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

std::tuple<TensorDict, torch::Tensor, torch::Tensor> ImpVecEnv::Step(const TensorDict &reply) {
  std::vector<TensorDict> obs_vector;
  torch::Tensor batch_reward = torch::zeros(envs_.size(), {torch::kFloat});
  torch::Tensor batch_terminal = torch::zeros(envs_.size(), {torch::kBool});
  TensorDict obs;
  float reward;
  bool terminal;
  for (size_t i = 0; i < envs_.size(); ++i) {
    auto rep = tensor_dict::Index(reply, i);
    std::tie(obs, reward, terminal) = envs_[i]->Step(rep);
    obs_vector.emplace_back(obs);
    batch_reward[i] = reward;
    batch_terminal[i] = terminal;
  }
  return std::make_tuple(tensor_dict::Stack(obs_vector, 0), batch_reward, batch_terminal);
}

bool ImpVecEnv::AnyTerminated() const {
  for (size_t i = 0; i < envs_.size(); i++) {
    if (envs_[i]->Terminated()) {
      return true;
    }
  }
  return false;
}

} // namespace rl::bridge