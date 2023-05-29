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
TensorDict VectorState::GetFeature() const {
  std::vector<TensorDict> obs(states_.size());
  for (size_t i = 0; i < states_.size(); ++i) {
    obs[i] = MakeObsTensorDict(states_[i], 1);
  }
  return rl::tensor_dict::Stack(obs, 0);
}
void VectorState::Display(int num_states) const {
  int num_display = std::min(Size(), num_states);
  for (size_t i = 0; i < num_display; ++i) {

    std::cout << "state " << i << ":\n" << states_[i] << std::endl;
  }
}
void VectorState::ApplyAction(const TensorDict &reply) {
  auto actions = reply.at("a");
  for (size_t i = 0; i < states_.size(); ++i) {
    if (!states_[i]->Terminated()) {
      states_[i]->ApplyAction(actions[static_cast<int>(i)].item<int>());
    }
  }
}
void VectorState::ApplyAction(const Action action) {
  for (size_t i = 0; i < states_.size(); ++i) {
    if (!states_[i]->Terminated()) {
      states_[i]->ApplyAction(action);
    }
  }
}
bool VectorState::AllTerminated() const {
  for (size_t i = 0; i < states_.size(); ++i) {
    if (!states_[i]->Terminated()) {
      return false;
    }
  }
  return true;
}
std::shared_ptr<VectorState> VectorState::Clone() const {
  auto vec_state = std::make_shared<VectorState>();
  for (size_t i = 0; i < states_.size(); ++i) {
    vec_state->Push(states_[i]->Clone());
  }
  return vec_state;
}
void VectorState::ComputeDoubleDummyResults() {
  std::vector<std::vector<Action>> cards_vector(states_.size());
  for (size_t i = 0; i < states_.size(); ++i) {
    std::cout << i << ", ";
    cards_vector[i] = states_[i]->GetCards();
  }
  std::cout << std::endl;
  std::vector<ddTableResults> table_results = CalcDDTs(cards_vector, -1);
  for (size_t i = 0; i < states_.size(); ++i) {
    std::cout << i << ", ";
    states_[i]->SetDoubleDummyResults(table_results[i]);
  }
}
std::vector<float> VectorState::Returns(Player player) {
  RL_CHECK_TRUE(AllTerminated())
  std::vector<float> ret(states_.size());
  for (size_t i = 0; i < states_.size(); ++i) {
    ret[i] = states_[i]->Returns()[player];
  }
  return ret;
}
void VectorState::ComputeDoubleDummyResultsSimple() {
  for (size_t i = 0; i < states_.size(); ++i) {
    std::cout << i << ", ";
    states_[i]->ComputeDoubleDummyResult();
  }
}
std::shared_ptr<VectorState> ParticleSampler::Sample(const std::shared_ptr<BridgeBiddingState> &state) {
  auto vec_state = std::make_shared<VectorState>();
  Player current_player = state->CurrentPlayer();
  auto player_cards = state->GetPlayerCards(current_player);
  for (size_t i = 0; i < batch_size_; ++i) {
    auto sample_cards = JointlySample(player_cards, current_player, rng_);
    BridgeDeal deal{sample_cards};
    auto new_state = std::make_shared<BridgeBiddingState>(deal);
    vec_state->Push(new_state);
  }
  return vec_state;
}
std::vector<int> Searcher::FilterParticles(const std::shared_ptr<VectorState> &vec_state,
                                           const std::vector<Action> &bid_history,
                                           const Player searching_player) {
  std::set<int> evicted;
  Player current_player = kNorth;
  std::vector<int> all_particle_indices = rl::utils::Arange(0, batch_size_);
  if (bid_history.empty()) {
    return all_particle_indices;
  }

  for (const auto bid : bid_history) {
    PrintIfVerbose("Current bid is ", BidString(bid), ".");
    if (current_player != searching_player) {
      auto obs = vec_state->GetFeature();
      TensorDict reply = actors_[current_player]->Act(obs);
      torch::Tensor probs = torch::exp(reply.at("log_probs"));
      auto [max_probs, max_indices] = probs.max(1);
      max_indices = max_indices.to(torch::kInt);
      torch::Tensor random_probs = torch::rand(batch_size_, {torch::kFloat});
      torch::Tensor action_probs = probs.index({torch::indexing::Slice(), bid});
      torch::Tensor less_indices = torch::nonzero(torch::le(action_probs, random_probs)).squeeze(1).to(torch::kInt);
//      torch::Tensor less_indices = torch::nonzero(torch::ne(max_indices, {bid})).squeeze(1).to(torch::kInt);
      int num_evicting = static_cast<int>(less_indices.numel());

      // some particles are filtered
      if (num_evicting >= 1) {
        for (int i = 0; i < num_evicting; ++i) {
          evicted.insert(less_indices[i].item<int>());
        }
        // all filtered
        if (evicted.size() == batch_size_) {
          return {};
        }
      }
    }
    vec_state->ApplyAction(bid);
    current_player = (current_player + 1) % kNumPlayers;
  }

  std::vector<int> remained_indices;
  std::set_difference(all_particle_indices.begin(), all_particle_indices.end(),
                      evicted.begin(), evicted.end(),
                      std::inserter(remained_indices, remained_indices.end()));
  return remained_indices;
}

std::vector<float> Searcher::Rollout(const std::shared_ptr<VectorState> &vec_state, Action bid, Player current_player) {
  auto cloned_vec_state = vec_state->Clone();
//  if (params_.verbose_level == kVerbose) {
//    cloned_vec_state->Display(5);
//  }
  cloned_vec_state->ApplyAction(bid);
  Player player = (current_player + 1) % kNumPlayers;
  while (!cloned_vec_state->AllTerminated()) {
    TensorDict obs = cloned_vec_state->GetFeature();
    TensorDict reply = actors_[player]->Act(obs);
    cloned_vec_state->ApplyAction(reply);
    player = (player + 1) % kNumPlayers;
  }
  return cloned_vec_state->Returns(current_player);
}

Action Searcher::Search(const std::shared_ptr<BridgeBiddingState> &state, const torch::Tensor &probs) {
  torch::Tensor top_k_actions, top_k_probs;
  std::tie(top_k_actions, top_k_probs) = GetTopKActions(probs, params_.top_k, params_.min_prob);
  PrintIfVerbose("Got top k actions and probs.");
  auto accessor = top_k_actions.accessor<int, 1>();
  int num_actions = static_cast<int>(top_k_actions.size(0));
  PrintIfVerbose(num_actions, " actions have prob > ", params_.min_prob, ":");
  if (params_.verbose_level == kVerbose){
    for (int i = 0; i < num_actions; ++i) {
      std::cout << BidString(accessor[i]) << " " << top_k_probs[i].item<float>() << std::endl;
    }
  }
  // only one action, no need to search
  if (num_actions == 1) {
    return top_k_actions[0].item<Action>();
  }
  // no action has a prob greater than min_prob
  if (num_actions == 0) {
    return torch::argmax(probs).item<Action>();
  }
  std::vector<Action> bid_history = state->BidHistory();
  PrintIfVerbose("Got bid history.");

  // no need to search if nobody ever bid.
//  if (bid_history.empty()){
//    return top_k_actions[0].item<Action>();
//  }
  int num_rollouts{}, num_particles{};
  torch::Tensor values = torch::zeros(num_actions, {torch::kFloat});
  auto vec_state_for_rollout = std::make_shared<VectorState>();
  PrintIfVerbose("Start filtering particles.");
  while (num_rollouts < params_.max_rollouts && num_particles < params_.max_particles) {
    // sample batch states
    std::shared_ptr<VectorState> vec_state = sampler_.Sample(state);
    std::vector<int> remained_particles = FilterParticles(vec_state, bid_history, state->CurrentPlayer());
    PrintIfVerbose("Number of remained particles: ", remained_particles.size());

    num_particles += batch_size_;
    if (!remained_particles.empty()) {
      auto states = vec_state->GetStates();
      for (const auto idx : remained_particles) {
        vec_state_for_rollout->Push(states[idx]);
        ++num_rollouts;
        if (vec_state_for_rollout->Size() == params_.max_rollouts) {
          break;
        }
      }
    }
  }
  PrintIfVerbose("Number of particles: ", num_particles);
  PrintIfVerbose("Number of rollouts: ", num_rollouts);
  // enough particles for rollouts or # sampled particles > max particles
  PrintIfVerbose("Starting rollouts.");

  if (num_particles >= params_.max_particles || vec_state_for_rollout->Size() >= params_.min_rollouts) {
    for (int i = 0; i < num_actions; ++i) {
      Action current_action = accessor[i];
      PrintIfVerbose("Action: ", BidString(current_action));
      auto rollout_values = Rollout(vec_state_for_rollout, current_action, state->CurrentPlayer());
      float sum_values = rl::utils::SumUpVector(rollout_values);
      values[i] += sum_values;
    }

  }
  // select action
  PrintIfVerbose("values:\n", values);
  PrintIfVerbose("Selecting actions.");

  // update the policy
  if (params_.select_highest_rollout_value) {
    if (num_rollouts > params_.min_rollouts) {
      PrintIfVerbose("num rollouts > min rollouts, return the action with highest value.");
      auto policy_posterior = top_k_probs * torch::exp(values / (params_.temperature * sqrt(num_rollouts)));
      auto greedy_action = top_k_actions[torch::argmax(policy_posterior, 0)].item<Action>();
      return greedy_action;
    } else {
      PrintIfVerbose("num rollouts < min rollouts, return the greedy action.");
      return torch::argmax(probs).item<Action>();
    }
  } else {
    if (num_rollouts > params_.min_rollouts) {
      // compute policy posterior following lockhart's paper
      torch::Tensor values_all_action = torch::zeros(kNumCalls, {torch::kFloat});
      values_all_action.scatter_(0, top_k_actions.to(torch::kLong), values);
      PrintIfVerbose("values for all actions\n", values_all_action);
      torch::Tensor
          policy_posterior = probs * torch::exp(values_all_action / (params_.temperature * sqrt(num_rollouts)));
      PrintIfVerbose("policy posterior:\n", policy_posterior);
      return torch::argmax(policy_posterior).item<Action>();
    } else {
      return torch::argmax(probs).item<Action>();
    }
  }
}
}