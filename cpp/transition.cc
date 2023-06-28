//
// Created by qzz on 2023/5/7.
//
#include "transition.h"
namespace rl::bridge {

Transition Transition::Index(int i) const {
  Transition elem;

  for (auto &name2tensor : obs) {
    elem.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto &name2tensor : reply) {
    elem.reply.insert({name2tensor.first, name2tensor.second[i]});
  }

  elem.reward = reward[i];
  elem.terminal = terminal[i];

  for (auto &name2tensor : next_obs) {
    elem.next_obs.insert({name2tensor.first, name2tensor.second[i]});
  }

  return elem;
}

Transition Transition::MakeBatch(const std::vector<Transition> &transitions, const std::string &device) {
  std::vector<TensorDict> obs_vec;
  std::vector<TensorDict> reply_vec;
  std::vector<torch::Tensor> reward_vec;
  std::vector<torch::Tensor> terminal_vec;
  std::vector<TensorDict> next_obs_vec;
  for (size_t i = 0; i < transitions.size(); ++i) {
    obs_vec.push_back(transitions[i].obs);
    reply_vec.push_back(transitions[i].reply);
    reward_vec.push_back(transitions[i].reward);
    terminal_vec.push_back(transitions[i].terminal);
    next_obs_vec.push_back(transitions[i].next_obs);
  }

  Transition batch;
  batch.obs = tensor_dict::Stack(obs_vec, 0);
  batch.reply = tensor_dict::Stack(reply_vec, 0);
  batch.reward = torch::stack(reward_vec, 0);
  batch.terminal = torch::stack(terminal_vec, 0);
  batch.next_obs = tensor_dict::Stack(next_obs_vec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto ToDevice = [&](const torch::Tensor &t) { return t.to(d); };
    batch.obs = tensor_dict::Apply(batch.obs, ToDevice);
    batch.reply = tensor_dict::Apply(batch.reply, ToDevice);
    batch.reward = batch.reward.to(d);
    batch.terminal = batch.terminal.to(d);
    batch.next_obs = tensor_dict::Apply(batch.next_obs, ToDevice);
  }
  return batch;
}
TensorDict Transition::ToDict() const {
  TensorDict ret = obs;
  for (auto &kv : next_obs) {
    ret.emplace("next_" + kv.first, kv.second);
  }

  for (auto &kv : reply) {
    ret.emplace(kv.first, kv.second);
  }

  ret.emplace("reward", reward);
  ret.emplace("terminal", terminal);
  return ret;
}

Transition Transition::SampleIllegalTransitions(float illegal_reward) const {
  torch::Tensor raw_probs = reply.at("raw_probs");
  auto d = raw_probs.device();
  auto ToDevice = [&](const torch::Tensor &t) { return t.to(d); };
  torch::Tensor legal_actions = obs.at("legal_actions");
  torch::Tensor illegal_actions_mask = 1 - legal_actions;
  torch::Tensor illegal_probs = illegal_actions_mask * raw_probs;
  torch::Tensor illegal_actions = torch::argmax(illegal_probs, 1);
  auto non_zero_indices = torch::nonzero(torch::any(illegal_probs != 0, 1)).squeeze(1);
  TensorDict illegal_obs = rl::tensor_dict::IndexSelect(obs, non_zero_indices);
  illegal_obs = rl::tensor_dict::Apply(illegal_obs, ToDevice);
  TensorDict illegal_reply = rl::tensor_dict::IndexSelect(reply, non_zero_indices);
  illegal_reply = rl::tensor_dict::Apply(illegal_reply, ToDevice);

  // replace illegal actions
  illegal_reply["a"] = illegal_actions;
  torch::Tensor illegal_rewards = torch::zeros(non_zero_indices.numel(), {torch::kFloat}).fill_(illegal_reward).to(d);
  torch::Tensor illegal_terminal = torch::zeros(non_zero_indices.numel(), {torch::kBool}).to(d);
  TensorDict illegal_next_obs = rl::tensor_dict::ZerosLike(illegal_obs);
  illegal_next_obs = rl::tensor_dict::Apply(illegal_next_obs, ToDevice);

  // merge them
  TensorDict ret_obs = rl::tensor_dict::BatchStack(obs, illegal_obs);
  TensorDict ret_reply = rl::tensor_dict::BatchStack(reply, illegal_reply);
  torch::Tensor ret_reward = torch::hstack({reward, illegal_rewards});
  torch::Tensor ret_terminal = torch::hstack({terminal, illegal_terminal});
  TensorDict ret_next_obs = rl::tensor_dict::BatchStack(next_obs, illegal_next_obs);

  Transition ret(ret_obs, ret_reply, ret_reward, ret_terminal, ret_next_obs);
  return ret;

}

SearchTransition SearchTransition::MakeBatch(const std::vector<SearchTransition> &transitions,
                                             const std::string &device) {
  std::vector<TensorDict> obs_vec;
  std::vector<torch::Tensor> policy_vec;
  std::vector<torch::Tensor> value_vec;
  for (const auto &t : transitions) {
    obs_vec.push_back(t.obs);
    policy_vec.push_back(t.policy_posterior);
    value_vec.push_back(t.value);
  }

  SearchTransition batch;
  batch.obs = rl::tensor_dict::Stack(obs_vec, 0);
  batch.policy_posterior = torch::vstack(policy_vec);
  batch.value = torch::hstack(value_vec);
  if (device != "cpu") {
    auto d = torch::Device(device);
    auto ToDevice = [&](const torch::Tensor &t) { return t.to(d); };
    batch.obs = tensor_dict::Apply(batch.obs, ToDevice);
    batch.policy_posterior = batch.policy_posterior.to(d);
    batch.value = batch.value.to(d);
  }
  return batch;
}

TensorDict SearchTransition::ToDict() const {
  TensorDict ret = obs;
  ret.emplace("policy_posterior", policy_posterior);
  ret.emplace("value", value);
  return ret;
}

ObsBelief ObsBelief::Index(int i) const {
  ObsBelief elem;

  for (auto &name2tensor : obs) {
    elem.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto &name2tensor : belief) {
    elem.belief.insert({name2tensor.first, name2tensor.second[i]});
  }

  return elem;
}
ObsBelief ObsBelief::MakeBatch(const std::vector<ObsBelief>& obs_beliefs, const std::string &device) {
  std::vector<TensorDict> obs_vec;
  std::vector<TensorDict> belief_vec;
  for (const auto &obs_belief : obs_beliefs) {
    obs_vec.push_back(obs_belief.obs);
    belief_vec.push_back(obs_belief.belief);
  }

  ObsBelief ret;
  ret.obs = tensor_dict::Stack(obs_vec, 0);
  ret.belief = tensor_dict::Stack(belief_vec, 0);
  if (device != "cpu") {
    auto d = torch::Device(device);
    auto ToDevice = [&](const torch::Tensor &t) { return t.to(d); };
    ret.obs = tensor_dict::Apply(ret.obs, ToDevice);
    ret.belief = tensor_dict::Apply(ret.belief, ToDevice);
  }
  return ret;
}
}