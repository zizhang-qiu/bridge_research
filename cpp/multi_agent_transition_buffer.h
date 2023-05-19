//
// Created by qzz on 2023/4/20.
//

#ifndef BRIDGE_RESEARCH_MULTI_AGENT_TRANSITION_BUFFER_H_
#define BRIDGE_RESEARCH_MULTI_AGENT_TRANSITION_BUFFER_H_
#include <vector>
#include "transition.h"
#include "replay_buffer.h"
#include "torch/torch.h"
namespace rl::bridge {

// A bridge transition buffer only stores
// obs, action and log probs, the sparse reward is given
// only at terminal as imp or duplicate score.
// So there is no need to store them all.
class BridgeTransitionBuffer {
 public:
  BridgeTransitionBuffer() = default;

  bool PushObsAndReply(const TensorDict &obs,
                       const TensorDict &reply);
  void Clear();

  [[nodiscard]] int Size() const { return static_cast<int>(obs_history_.size()); }

  void PushToReplayBuffer(std::shared_ptr<Replay> &replay_buffer, float final_reward) const;

  [[nodiscard]] std::tuple<std::vector<Transition>, torch::Tensor> PopTransitions(float final_reward) const;

 private:
  std::vector<TensorDict> obs_history_;
  std::vector<TensorDict> reply_history_;
};

class MultiAgentTransitionBuffer {
 public:

  explicit MultiAgentTransitionBuffer(int num_agents);
  void PushObsAndReply(int player,
                       const TensorDict &obs,
                       const TensorDict &reply);
  void PushToReplayBuffer(std::shared_ptr<Replay> replay_buffer, const std::vector<float> &reward);
  void Clear();

 private:
  int num_agents_;
  std::vector<BridgeTransitionBuffer> storage_;
};
}//namespace rl::bridge
#endif //BRIDGE_RESEARCH_MULTI_AGENT_TRANSITION_BUFFER_H_
