//
// Created by qzz on 2023/6/28.
//
#include "search.h"
namespace rl::bridge {

std::vector<std::vector<Action>> BeliefModel::Sample(const TensorDict &obs, Player current_player, int num_sample) {
  std::vector<std::vector<Action>> deals;
  torch::NoGradGuard ng;
  TorchJitInput input;
  int id = -1;
  auto model = model_locker_->GetModel(&id);
  input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
  input.emplace_back(current_player);
  input.emplace_back(num_sample);
  auto output = model.get_method("sample")(input);
  auto reply = tensor_dict::FromIValue(output, torch::kCPU, true);
  torch::Tensor sampled_deals = reply.at("cards");
  for (int64_t i = 0; i < num_sample; ++i) {
    torch::Tensor cards_tensor = sampled_deals[i].clone();
    cards_tensor.contiguous();
    std::vector<Action> cards(cards_tensor.data_ptr<int>(), cards_tensor.data_ptr<int>() + cards_tensor.numel());
    deals.push_back(cards);
  }
  return deals;
}
TensorDict VectorState::GetFeature() const {
  std::vector<TensorDict> obs_vec;
  obs_vec.reserve(states_.size());
  for (const auto &state : states_) {
    obs_vec.push_back(MakeObsTensorDict(state, 1));
  }
  return tensor_dict::Stack(obs_vec, 0);
}
void VectorState::Step(const TensorDict &reply) {
  for (size_t i = 0; i < states_.size(); ++i) {
    if (!states_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      states_[i]->ApplyAction(rep.at("a").item<Action>());
    }
  }
}
bool VectorState::AllTerminated() const {
  for (const auto &state : states_) {
    if (!state->Terminated()) {
      return false;
    }
  }
  return true;
}
void VectorState::Step(const Action bid) {
  for (size_t i = 0; i < states_.size(); ++i) {
    states_[i]->ApplyAction(bid);
  }
}
std::shared_ptr<VectorState> VectorState::Clone() const{
  auto cloned = std::make_shared<VectorState>();
  for (size_t i = 0; i < states_.size(); ++i) {
    cloned->Push(states_[i]->Clone());
  }
  return cloned;
}
std::vector<double> VectorState::GetReturns(Player player) const {
  RL_CHECK_TRUE(AllTerminated())
  std::vector<double> returns;
  for (size_t i = 0; i < states_.size(); ++i){
    returns.push_back(states_[i]->Returns()[player]);
  }
  return returns;
}
std::vector<double> Rollout(const std::shared_ptr<VectorState>& vec_state,
                            const std::shared_ptr<VecEnvActor>& actor,
                            const Action bid, const Player player) {
  auto cloned = vec_state->Clone();
  cloned->Step(bid);
  TensorDict obs, reply;
  while(!cloned->AllTerminated()){
    obs = cloned->GetFeature();
    reply = actor->Act(obs);
    cloned->Step(reply);
  }
  return cloned->GetReturns(player);
}
std::tuple<torch::Tensor, torch::Tensor> GetTopKActions(const torch::Tensor &probs, const int k, const float min_prob) {
  auto probs_ = probs.to(torch::Device("cpu"));
  torch::Tensor top_k_probs, top_k_indices;
  std::tie(top_k_probs, top_k_indices) = torch::topk(probs_, k);
  torch::Tensor available_indices = (top_k_probs > min_prob);
  return std::make_tuple(top_k_indices.masked_select(available_indices).to(torch::kInt),
                         top_k_probs.masked_select(available_indices));
}
}