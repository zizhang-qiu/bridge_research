//
// Created by qzz on 2023/4/29.
//

#ifndef BRIDGE_RESEARCH_CPP_SEARCH_H_
#define BRIDGE_RESEARCH_CPP_SEARCH_H_
#include "bridge_actor.h"
#include "bridge_state.h"
#include "bridge_envs.h"
#include "utils.h"

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
  int verbose_level = silent;
};

std::vector<Action> JointlySample(const std::vector<Action> &cards, Player player) {
  std::vector<Action> ret;
  std::set_difference(all_cards.begin(), all_cards.end(),
                      cards.begin(), cards.end(),
                      std::inserter(ret, ret.end()));
  std::shuffle(ret.begin(), ret.end(), std::mt19937(std::random_device()()));
  for (int i = 0; i < kNumCardsPerHand; ++i) {
    ret.insert(ret.begin() + player + i * kNumPlayers, cards[i]);
  }
  return ret;
}

std::shared_ptr<BridgeBiddingState> SampleParticle(const std::vector<Action> cards,
                                                   Player current_player) {
  auto ret = JointlySample(cards, current_player);
  BridgeDeal deal{ret, kNorth, false, false};
  auto ret_state = std::make_shared<BridgeBiddingState>(deal);
  return ret_state;
}

double RolloutValue(const std::shared_ptr<BridgeBiddingState> &state,
                    const Action bid, const std::shared_ptr<SingleEnvActor> &actor,
                    const Player current_player) {
  auto cloned_state = state->Clone();
  cloned_state->ApplyAction(bid);
  TensorDict obs;
  while (!cloned_state->Terminated()) {
    obs = MakeObsTensorDict(cloned_state, 1);
    auto reply = actor->Act(obs);
    auto action = reply.at("a").item<Action>();
    cloned_state->ApplyAction(action);
  }
  auto returns = cloned_state->Returns();
  return returns[current_player];
}

Action Search(const std::shared_ptr<BridgeBiddingState> &state,
              const std::shared_ptr<SingleEnvActor> &actor,
              const SearchParams params) {
  Player current_player = state->CurrentPlayer();
  auto player_cards = state->GetPlayerCards(current_player);
  std::vector<Action> bid_history = state->BidHistory();
  if (params.verbose_level) {
    std::cout << "Get bid history." << std::endl;
  }

  auto obs = MakeObsTensorDict(state, 1);
  auto reply = actor->GetTopKActionsWithMinProb(obs, params.top_k, params.min_prob);
  torch::Tensor top_k_actions = reply.at("top_k_actions");
  torch::Tensor top_k_probs = reply.at("top_k_probs");
  if (params.verbose_level) {
    std::cout << "top k actions: " << top_k_actions << std::endl;
    std::cout << "top k probs: " << top_k_probs << std::endl;
  }
  int num_actions = static_cast<int>(top_k_actions.size(0));
  // only one action, no need to search
  if (num_actions == 1) {
    return top_k_actions[0].item<Action>();
  }
  int num_rollouts = 0;
  int num_particles = 0;
  torch::Tensor values = torch::zeros(num_actions, {torch::kFloat});
  if (params.verbose_level) {
    std::cout << "Create value tensor." << std::endl;
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  if (params.verbose_level) {
    std::cout << "Sample particle." << std::endl;
  }
  while (num_particles < params.max_particles && num_rollouts < params.max_rollouts) {
    if (params.verbose_level) {
      std::cout << "\rnum_particles: " << num_particles << "num_rollouts: " << num_rollouts;
    }
    auto particle = SampleParticle(player_cards, current_player);
    ++num_particles;

    // filter particles
    bool skip = false;
    for (const auto bid : bid_history) {
      auto obs_ = MakeObsTensorDict(particle, 1);
      auto prob = actor->GetProbForAction(obs_, bid);
      double random_number = dist(rng);
      if (prob < random_number) {
        skip = true;
        break;
      }
      particle->ApplyAction(bid);
    }
    if (skip) {
      continue;
    }

    // update value

    auto accessor = top_k_actions.accessor<int, 1>();
    for (int i = 0; i < num_actions; i++) {
      auto action = accessor[i];
      double rollout_value = RolloutValue(particle, action, actor, current_player);
      values[i] += rollout_value;
    }

    ++num_rollouts;

  }
  if (params.verbose_level) {
    std::cout << "\nnum_particles: " << num_particles << "  , num_rollouts: " << num_rollouts << std::endl;
    std::cout << values << std::endl;
//      std::cout << particle << std::endl;
  }

  if (params.verbose_level) {
    std::cout << "Get greedy action." << std::endl;
  }
  torch::Tensor probs_posterior;
  if (num_rollouts > params.min_rollouts) {
    probs_posterior = top_k_probs * torch::exp(values / (params.temperature * sqrt(num_rollouts)));
  } else {
    probs_posterior = top_k_probs;
  }
  if (params.verbose_level) {
    std::cout << "probs posterior: " << probs_posterior << std::endl;
  }
  auto greedy_action = top_k_actions[torch::argmax(probs_posterior)].item<Action>();
  return greedy_action;
}
}
#endif //BRIDGE_RESEARCH_CPP_SEARCH_H_
