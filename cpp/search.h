//
// Created by qzz on 2023/4/29.
//

#ifndef BRIDGE_RESEARCH_CPP_SEARCH_H_
#define BRIDGE_RESEARCH_CPP_SEARCH_H_
#include "bridge_actor.h"
#include "bridge_state.h"
#include "bridge_envs.h"
#include "rl/utils.h"

#include <algorithm>

namespace rl::bridge {

const std::vector<Action> all_cards = utils::Arange(0, kNumCards);
enum VerboseLevel {
  silent = 0, verbose
};

struct SearchParams {
  int min_rollouts = 100;
  int max_rollouts = 1000;
  int max_particles = 100000;
  float temperature = 100.0f;
  int top_k = 4;
  float min_prob = 1e-4f;
  int seed = 0;
  int verbose_level = silent;
};

std::vector<Action> JointlySample(const std::vector<Action> &cards, Player player, std::mt19937 &rng);

std::shared_ptr<BridgeBiddingState> SampleParticle(const std::vector<Action> &cards,
                                                   Player current_player,
                                                   std::mt19937 &rng);
// todo: use four actors
double RolloutValue(const std::shared_ptr<BridgeBiddingState> &state,
                    Action bid, const std::shared_ptr<SingleEnvActor> &actor,
                    Player current_player);

std::tuple<torch::Tensor, torch::Tensor> GetTopKActions(const torch::Tensor &probs, int k, float min_prob);

Action Search(const torch::Tensor &probs,
              const std::shared_ptr<BridgeBiddingState> &state,
              const std::shared_ptr<SingleEnvActor> &actor,
              SearchParams params);
}
#endif //BRIDGE_RESEARCH_CPP_SEARCH_H_
