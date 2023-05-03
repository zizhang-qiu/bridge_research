//
// Created by qzz on 2023/4/20.
//

#ifndef BRIDGE_RESEARCH_MULTI_AGENT_TRANSITION_BUFFER_H_
#define BRIDGE_RESEARCH_MULTI_AGENT_TRANSITION_BUFFER_H_
#include <vector>
#include "torch/torch.h"
#include "replay_buffer.h"
namespace rl::bridge {

// A bridge transition buffer only stores
// obs, action and log probs, the sparse reward is given
// only at terminal as imp or duplicate score.
// So there is no need to store them all.
class BridgeTransitionBuffer {
 public:
  BridgeTransitionBuffer() = default;

  bool PushObsActionLogProbs(const torch::Tensor &obs,
                             const torch::Tensor &action,
                             const torch::Tensor &log_probs);
  void Clear();
  void PushToReplayBuffer(std::shared_ptr<ReplayBuffer> &replay_buffer, double final_reward);

 private:
  std::vector<torch::Tensor> obs_history_;
  std::vector<torch::Tensor> action_history_;
  std::vector<torch::Tensor> log_probs_history_;
};

class MultiAgentTransitionBuffer {
 public:

  explicit MultiAgentTransitionBuffer(int num_agents);
  void PushObsActionLogProbs(int player,
                             const torch::Tensor &obs,
                             const torch::Tensor &action,
                             const torch::Tensor &log_probs);
  void PushToReplayBuffer(std::shared_ptr<ReplayBuffer> replay_buffer, const std::vector<double> &reward);
  void Clear();

 private:
  int num_agents_;
  std::vector<BridgeTransitionBuffer> storage_;
};
}//namespace rl::bridge
#endif //BRIDGE_RESEARCH_MULTI_AGENT_TRANSITION_BUFFER_H_
