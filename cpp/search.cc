//
// Created by qzz on 2023/5/3.
//
#include "search.h"
namespace rl::bridge {
std::vector<Action> JointlySample(const std::vector<Action> &cards, const Player player, std::mt19937 &rng) {
  std::vector<Action> ret;
  std::set_difference(all_cards.begin(), all_cards.end(),
                      cards.begin(), cards.end(),
                      std::inserter(ret, ret.end()));
  std::shuffle(ret.begin(), ret.end(), rng);
  for (int i = 0; i < kNumCardsPerHand; ++i) {
    ret.insert(ret.begin() + player + i * kNumPlayers, cards[i]);
  }
  return ret;
}

std::shared_ptr<BridgeBiddingState> SampleParticle(const std::vector<Action> &cards,
                                                   const Player current_player,
                                                   std::mt19937 &rng) {
  auto ret = JointlySample(cards, current_player, rng);
  BridgeDeal deal{ret, kNorth, false, false};
  auto ret_state = std::make_shared<BridgeBiddingState>(deal);
  return ret_state;
}

double RolloutValue(const std::shared_ptr<BridgeBiddingState> &state,
                    const Action bid,
                    const std::vector<std::shared_ptr<SingleEnvActor>> &actors,
                    const Player current_player) {
  auto cloned_state = state->Clone();
  cloned_state->ApplyAction(bid);
  TensorDict obs;
  while (!cloned_state->Terminated()) {
    obs = MakeObsTensorDict(cloned_state, 1);
    Player acting_player = cloned_state->CurrentPlayer();
    auto reply = actors[acting_player]->Act(obs);
    auto action = reply.at("a").item<Action>();
    cloned_state->ApplyAction(action);
  }
  auto returns = cloned_state->Returns();
  return returns[current_player];
}

std::tuple<torch::Tensor, torch::Tensor> GetTopKActions(const torch::Tensor &probs,
                                                        const int k,
                                                        const float min_prob) {
  auto probs_ = probs.to(torch::Device("cpu"));
  torch::Tensor top_k_probs, top_k_indices;
  std::tie(top_k_probs, top_k_indices) = torch::topk(probs_, k);
  torch::Tensor available_indices = (top_k_probs > min_prob);
  return std::make_tuple(top_k_indices.masked_select(available_indices).to(torch::kInt),
                         top_k_probs.masked_select(available_indices));
}

Action Search(const torch::Tensor &probs,
              const std::shared_ptr<BridgeBiddingState> &state,
              const std::vector<std::shared_ptr<SingleEnvActor>> &actors,
              const SearchParams params) {
  Player current_player = state->CurrentPlayer();
  auto player_cards = state->GetPlayerCards(current_player);
  std::vector<Action> bid_history = state->BidHistory();
  if (params.verbose_level == kVerbose) {
    std::cout << "Get bid history." << std::endl;
  }

  torch::Tensor top_k_actions, top_k_probs;
  std::tie(top_k_actions, top_k_probs) = GetTopKActions(probs, params.top_k, params.min_prob);
  auto accessor = top_k_actions.accessor<int, 1>();
  if (params.verbose_level == kVerbose) {
    std::cout << "top k actions: " << top_k_actions << std::endl;
    std::cout << "top k probs: " << top_k_probs << std::endl;
  }
  int num_actions = static_cast<int>(top_k_actions.size(0));
  if (params.verbose_level == kVerbose) {
    std::cout << "num_actions: " << num_actions << std::endl;
  }
  // only one action, no need to search
  if (num_actions == 1) {
    return top_k_actions[0].item<Action>();
  }
  // no action has a prob greater than min_prob
  if (num_actions == 0) {
    return torch::argmax(probs).item<Action>();
  }
  int num_rollouts = 0;
  int num_particles = 0;
  torch::Tensor values = torch::zeros(num_actions, {torch::kFloat});
  if (params.verbose_level == kVerbose) {
    std::cout << "Create value tensor." << std::endl;
  }

  std::mt19937 rng;
  rng.seed(params.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  if (params.verbose_level == kVerbose) {
    std::cout << "Sample particle." << std::endl;
  }
  while (num_particles < params.max_particles && num_rollouts < params.max_rollouts) {
    if (params.verbose_level) {
      std::cout << "\rnum_particles: " << num_particles << "  num_rollouts: " << num_rollouts;
    }
    auto particle = SampleParticle(player_cards, current_player, rng);
    ++num_particles;

    // filter particles
    bool skip = false;
    for (const auto bid : bid_history) {
      auto obs_ = MakeObsTensorDict(particle, 1);
      int acting_player = particle->CurrentPlayer();
      auto prob = actors[acting_player]->GetProbForAction(obs_, bid);
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

    particle->ComputeDoubleDummyResult();
    for (int i = 0; i < num_actions; i++) {
      auto action = accessor[i];
      double rollout_value = RolloutValue(particle, action, actors, current_player);
      values[i] += rollout_value;
    }

    ++num_rollouts;

  }
  if (params.verbose_level == kVerbose) {
    std::cout << "\nnum_particles: " << num_particles << "  , num_rollouts: " << num_rollouts << std::endl;
    std::cout << "values:\n" << values << std::endl;
//      std::cout << particle << std::endl;
  }

  if (params.verbose_level == kVerbose) {
    std::cout << "Get greedy action." << std::endl;
  }
//  torch::Tensor probs_posterior;
//  if (num_rollouts > params.min_rollouts) {
//    probs_posterior = top_k_probs * torch::exp(values / (params.temperature * sqrt(num_rollouts)));
//  } else {
//    probs_posterior = top_k_probs;
//  }
//  if (params.verbose_level == kVerbose) {
//    std::cout << "probs posterior: " << probs_posterior << std::endl;
//  }
  if (num_rollouts > params.min_rollouts) {
    auto greedy_action = top_k_actions[torch::argmax(values, 0)].item<Action>();
    return greedy_action;
  } else {
    return torch::argmax(probs).item<Action>();
  }
}
}