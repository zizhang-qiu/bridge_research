//
// Created by qzz on 2023/6/28.
//

#ifndef BRIDGE_RESEARCH_CPP_SEARCH_H_
#define BRIDGE_RESEARCH_CPP_SEARCH_H_
#include <utility>

#include "model_locker.h"
#include "tensor_dict.h"
#include "bridge_state.h"
#include "bridge_envs.h"
#include "bridge_actor.h"
namespace rl::bridge {
class BeliefModel {
 public:
  explicit BeliefModel(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)) {}

  std::vector<std::vector<Action>> Sample(const TensorDict &obs, Player current_player, int num_sample);
 private:
  std::shared_ptr<ModelLocker> model_locker_;
};

class VectorState {
 public:
  VectorState() = default;

  [[nodiscard]] std::shared_ptr<VectorState> Clone() const;

  void Push(const std::shared_ptr<BridgeBiddingState> &state) { states_.push_back(state); };
  [[nodiscard]] TensorDict GetFeature() const;

  void Step(const TensorDict &reply);

  void Step(Action bid);

  std::vector<double> GetReturns(Player player) const;

  [[nodiscard]] bool AllTerminated() const;
 private:
  std::vector<std::shared_ptr<BridgeBiddingState>> states_;
};

struct SearchParams {
  int num_particles = 1000;
  double temperature = 100.0;
  int top_k = 4;
  float min_prob = 1e-4f;
  bool verbose = false;
};

std::vector<double> Rollout(const std::shared_ptr<VectorState> &vec_state,
                            const std::shared_ptr<VecEnvActor> &actor,
                            Action bid, Player player);

std::tuple<torch::Tensor, torch::Tensor> GetTopKActions(const torch::Tensor &probs,
                                                        int k,
                                                        float min_prob);

class Searcher {
 public:
  Searcher(SearchParams params,
           std::shared_ptr<BeliefModel> belief_model,
           std::shared_ptr<VecEnvActor> actor)
      : params_(params),
        belief_model_(std::move(belief_model)),
        actor_(std::move(actor)) {}

  TensorDict Search(const std::shared_ptr<BridgeBiddingState> &state, const TensorDict &obs, const TensorDict &reply) {
    std::vector<int> bid_history = state->BidHistory();
    Player current_player = state->CurrentPlayer();
    // Get top actions
    auto probs = reply.at("raw_probs");
    probs /= torch::sum(probs);
    auto [top_k_actions, top_k_probs] = GetTopKActions(probs, params_.top_k, params_.min_prob);
    top_k_probs = top_k_probs.to(torch::kFloat64);
    int num_actions = static_cast<int>(top_k_actions.numel());
    if (params_.verbose) {
      std::cout << "Get top k actions:\n";
      for (int i = 0; i < num_actions; ++i) {
        std::cout << BidString(top_k_actions[i].item<int>()) << ":" << top_k_probs[i].item<double>() << std::endl;
      }
    }
    auto sampled_deals = belief_model_->Sample(obs, state->CurrentPlayer(), params_.num_particles);
    if (params_.verbose) {
      std::cout << "Sample deals." << std::endl;
    }
    // Build deals
    auto vec_state = std::make_shared<VectorState>();
    for (size_t i = 0; i < params_.num_particles; ++i) {
      const BridgeDeal deal{sampled_deals[i]};
      auto particle = std::make_shared<BridgeBiddingState>(deal);
      vec_state->Push(particle);
    }
    for (const auto &bid : bid_history) {
      vec_state->Step(bid);
    }
    if (params_.verbose) {
      std::cout << "Build deals." << std::endl;
    }

    // Rollout
    torch::Tensor values = torch::zeros(num_actions, {torch::kFloat64});
    for (int i = 0; i < num_actions; ++i) {
      if (params_.verbose){
        std::cout <<"Rollout action" << top_k_actions[i].item<Action>() << std::endl;
      }
      auto rollout_scores = Rollout(vec_state, actor_, top_k_actions[i].item<Action>(), current_player);
      values[i] = utils::SumUpVector(rollout_scores);
    }
    if (params_.verbose){
      std::cout << "Rollout values:\n";
      for(int i=0; i<num_actions; ++i){
        std::cout << BidString(top_k_actions[i].item<int>()) << ":" << values[i].item<double>() << std::endl;
      }
    }

    // Get policy
    torch::Tensor
        altered_probs = top_k_probs * torch::exp(values / (params_.temperature * sqrt(params_.num_particles)));
    torch::Tensor policy_posterior = torch::zeros(kNumCalls, {torch::kFloat64});
    policy_posterior.scatter_(0, top_k_actions.to(torch::kLong), altered_probs);
    auto action = policy_posterior.argmax();
    return {
        {"policy_posterior", policy_posterior},
        {"a", action}
    };
  }
 private:
  const SearchParams params_;
  std::shared_ptr<BeliefModel> belief_model_;
  std::shared_ptr<VecEnvActor> actor_;
};
}
#endif //BRIDGE_RESEARCH_CPP_SEARCH_H_
