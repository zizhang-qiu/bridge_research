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

}