//
// Created by qzz on 2023/4/29.
//

#ifndef BRIDGE_RESEARCH_CPP_SEARCH_H_
#define BRIDGE_RESEARCH_CPP_SEARCH_H_
#include "bridge_actor.h"
#include "bridge_state.h"
#include "bridge_envs.h"
#include "dds.h"
#include "rl/utils.h"
#include "torch/torch.h"

#include <algorithm>
#include <utility>

namespace rl::bridge {

const std::vector<Action> all_cards = utils::Arange(0, kNumCards);
enum VerboseLevel {
  kSilent = 0, kVerbose
};

struct SearchParams {
  int min_rollouts = 100;
  int max_rollouts = 1000;
  int max_particles = 100000;
  float temperature = 100.0f;
  int top_k = 4;
  float min_prob = 1e-4f;
  int seed = 1;
  int verbose_level = kSilent;
  // if False, select by highest probability in amended policy
  bool select_highest_rollout_value = true;
};

std::vector<Action> JointlySample(const std::vector<Action> &cards, Player player, std::mt19937 &rng);

std::shared_ptr<BridgeBiddingState> SampleParticle(const std::vector<Action> &cards,
                                                   Player current_player,
                                                   std::mt19937 &rng);
double RolloutValue(const std::shared_ptr<BridgeBiddingState> &state,
                    Action bid,
                    const std::vector<std::shared_ptr<SingleEnvActor>> &actors,
                    Player current_player);

std::tuple<torch::Tensor, torch::Tensor> GetTopKActions(const torch::Tensor &probs, int k, float min_prob);

Action Search(const torch::Tensor &probs,
              const std::shared_ptr<BridgeBiddingState> &state,
              const std::vector<std::shared_ptr<SingleEnvActor>> &actors,
              SearchParams params);

// a vector stores states, like vector env.
class VectorState {
 public:
  VectorState() = default;

  void Push(const std::shared_ptr<BridgeBiddingState> &state) {
    states_.push_back(state);
  }

  [[nodiscard]] TensorDict GetFeature() const;

  void Display(int num_states) const;

  // use this when rollout
  void ApplyAction(const TensorDict &reply);

  // use this when filter
  void ApplyAction(const Action action);

  void ComputeDoubleDummyResultsSimple();

  [[nodiscard]] bool AllTerminated() const;

  [[nodiscard]] std::shared_ptr<VectorState> Clone() const;

  void ComputeDoubleDummyResults();

  std::vector<float> Returns(Player player);

  [[nodiscard]] int Size() const { return static_cast<int>(states_.size()); }

  [[nodiscard]] std::vector<std::shared_ptr<BridgeBiddingState>> GetStates() const { return states_; }

 private:
  std::vector<std::shared_ptr<BridgeBiddingState>> states_;
};

class ParticleSampler {
 public:
  ParticleSampler(int seed, int batch_size)
      : rng_(seed),
        batch_size_(batch_size) {}

  std::shared_ptr<VectorState> Sample(const std::shared_ptr<BridgeBiddingState> &state);

 private:
  std::mt19937 rng_;
  int batch_size_;
};

class Searcher {
 public:
  Searcher(SearchParams params, std::vector<std::shared_ptr<VecEnvActor>> actors, int batch_size)
      : params_(params),
        actors_(std::move(actors)),
        batch_size_(batch_size),
        sampler_(params.seed, batch_size) {
    RL_CHECK_EQ(actors_.size(), kNumPlayers);
  }

  std::vector<int> FilterParticles(const std::shared_ptr<VectorState> &vec_state,
                                   const std::vector<Action> &bid_history,
                                   Player searching_player);

  std::vector<float> Rollout(const std::shared_ptr<VectorState> &vec_state,
                             Action bid, Player current_player);

  Action Search(const std::shared_ptr<BridgeBiddingState> &state, const torch::Tensor &probs);

 private:
  const SearchParams params_;
  const int batch_size_;
  std::vector<std::shared_ptr<VecEnvActor>> actors_;
  ParticleSampler sampler_;

  template<typename... Args>
  void PrintIfVerbose(const Args &... items) {
    if (params_.verbose_level == kVerbose) {
      (std::cout << ... << items) << std::endl;
    }
  }
};
}
#endif //BRIDGE_RESEARCH_CPP_SEARCH_H_
