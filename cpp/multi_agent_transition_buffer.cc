//
// Created by qzz on 2023/5/3.
//
#include "multi_agent_transition_buffer.h"
#include "rl/logging.h"
#include "rl/utils.h"
namespace rl::bridge {

bool BridgeTransitionBuffer::PushObsAndReply(const TensorDict &obs,
                                             const TensorDict &reply) {
  RL_CHECK_EQ(obs_history_.size(), reply_history_.size());
  bool available = utils::CheckProbNotZero(reply.at("a"), reply.at("log_probs"));
  if (available) {
    obs_history_.emplace_back(obs);
    reply_history_.emplace_back(reply);
  }
  return available;
}

void BridgeTransitionBuffer::Clear() {
  obs_history_.clear();
  reply_history_.clear();
}


std::tuple<std::vector<Transition>, torch::Tensor> BridgeTransitionBuffer::PopTransitions(const float final_reward) const {
  int size = Size();
  std::vector<Transition> transitions(size);
  torch::Tensor weights = torch::zeros({size}, torch::kFloat32);
  auto weight_acc = weights.accessor<float, 1>();
  for (int i = size - 1; i >= 0; --i) {
    Transition t;
    t.obs = obs_history_[i];
    t.reply = reply_history_[i];
    float td_error;
    t.reward = torch::tensor(final_reward, torch::kFloat32);
    if (i == size - 1) {
      t.terminal = torch::tensor(1, torch::kBool);
      t.next_obs = tensor_dict::ZerosLike(t.obs);

    } else {
      t.terminal = torch::tensor(0, torch::kBool);
      t.next_obs = obs_history_[i + 1];
    }
    td_error = final_reward - t.reply.at("values").item<float>();
    transitions[i] = t;
    weight_acc[i] = std::abs(td_error);
  }
  return std::make_tuple(transitions, weights);
}


void BridgeTransitionBuffer::PushToReplayBuffer(std::shared_ptr<Replay> &replay_buffer, const float final_reward) const {
  std::vector<Transition> transitions;
  torch::Tensor weights;
  std::tie(transitions, weights) = PopTransitions(final_reward);
  replay_buffer->Add(transitions, weights);
}


MultiAgentTransitionBuffer::MultiAgentTransitionBuffer(int num_agents)
    : num_agents_(num_agents) {
  for (size_t i = 0; i < num_agents_; i++) {
    storage_.emplace_back();
  }
}

void MultiAgentTransitionBuffer::PushObsAndReply(const int player,
                                                 const TensorDict &obs,
                                                 const TensorDict &reply) {
  storage_[player].PushObsAndReply(obs, reply);
}

void MultiAgentTransitionBuffer::PushToReplayBuffer(std::shared_ptr<Replay> replay_buffer,
                                                    const std::vector<float> &reward) {
  RL_CHECK_EQ(reward.size(), num_agents_);
  for (size_t pl = 0; pl < num_agents_; ++pl) {
    storage_[pl].PushToReplayBuffer(replay_buffer, reward[pl]);
  }
}

void MultiAgentTransitionBuffer::Clear() {
  for (size_t pl = 0; pl < num_agents_; ++pl) {
    storage_[pl].Clear();
  }
}

}