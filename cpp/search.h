//
// Created by qzz on 2023/6/28.
//

#ifndef BRIDGE_RESEARCH_CPP_SEARCH_H_
#define BRIDGE_RESEARCH_CPP_SEARCH_H_
#include <utility>

#include "model_locker.h"
#include "tensor_dict.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_envs.h"
#include "bridge_actor.h"
namespace rl::bridge {

bool CheckDealLegality(const Cards &cards);

class BeliefModel {
 public:
  explicit BeliefModel(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)) {}

  torch::Tensor GetBelief(const TensorDict &obs);
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

  void Pop(int num_pop);

  std::vector<int> ContractIndices() const;

  TensorDict GetFinalObservation() const;

  [[nodiscard]] int Size() const { return static_cast<int>(states_.size()); };

  [[nodiscard]] std::vector<double> GetReturns(Player player) const;

  std::vector<std::vector<double>> ReturnsByContractIndices(const std::vector<std::vector<int>> &indices,
                                                            Player player) const;

  [[nodiscard]] bool AllTerminated() const;

  std::vector<std::shared_ptr<BridgeBiddingState>> GetStates() const { return states_; }

  void Show(int num_show) const {
    num_show = std::min(num_show, Size());
    for (size_t i = 0; i < num_show; ++i) {
      std::cout << states_[i]->ToString() << std::endl;
    }
  }
 private:
  std::vector<std::shared_ptr<BridgeBiddingState>> states_;
};

class ScorePredictor {
 public:
  explicit ScorePredictor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)) {}

  std::vector<double> Predict(const std::shared_ptr<VectorState> &vec_state, Player player);
 private:
  std::shared_ptr<ModelLocker> model_locker_;
};

Cards Sample(const torch::Tensor &belief_pred, const TensorDict &obs, Player current_player);

Cards RandomSample(const std::shared_ptr<BridgeBiddingState> &state, Player player);

struct SearchParams {
  double prob_exponent = 1.0;
  double length_exponent = 0.5;
  int filter_batch_size = 1000;
  int max_try = 1000;
  double temperature = 100.0;
  int top_k = 4;
  int max_rollouts = 1000;
  int min_rollouts = 100;
  int max_particles = 100000;
  float min_prob = 1e-4f;
  float max_prob = 0.7f;
  bool random_sample = false;
  bool verbose = false;
};

std::vector<int> Rollout(const std::shared_ptr<VectorState> &vec_state,
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

  std::vector<int> FilterParticles(const std::vector<Cards> &cards,
                                   const std::vector<Action> &bid_history,
                                   Player current_player) {
    const int num_sample = static_cast<int>(cards.size());
    auto vec_state = std::make_shared<VectorState>();
    std::set<int> evicted;
    std::vector<int> all_indices = rl::utils::Arange(0, num_sample);
    if (bid_history.empty()) {
      return all_indices;
    }

    for (size_t i = 0; i < num_sample; ++i) {
      const BridgeDeal deal{cards[i]};
      auto particle = std::make_shared<BridgeBiddingState>(deal);
      vec_state->Push(particle);
    }
    TensorDict obs, reply;
    torch::Tensor probs, action_prob, random_numbers, less_indices;
    Player player = kNorth;
    // filter
    for (const auto bid : bid_history) {
      if (player != current_player) {
        obs = vec_state->GetFeature();
        reply = actor_->Act(obs);
        probs = reply.at("raw_probs") * obs.at("legal_actions");
        probs /= probs.sum(1, true);
        action_prob = probs.index({torch::indexing::Slice(), bid});

        random_numbers = torch::rand(num_sample, {torch::kFloat32});
        less_indices = torch::nonzero(torch::lt(action_prob, random_numbers)).squeeze(1).to(torch::kInt);
//        std::cout << "less indices:\n" << less_indices << std::endl;
        if (less_indices.numel() > 0) {
          auto less_indices_accessor = less_indices.accessor<int, 1>();
          for (int i = 0; i < less_indices.numel(); ++i) {
            evicted.insert(less_indices_accessor[i]);
          }
          if (evicted.size() == num_sample) {
            return {};
          }
        }
      }
      vec_state->Step(bid);
      player = (player + 1) % kNumPlayers;
    }
    std::vector<int> remained_indices;
    std::set_difference(all_indices.begin(), all_indices.end(),
                        evicted.begin(), evicted.end(), std::back_inserter(remained_indices));
    return remained_indices;
  }

  TensorDict Search(const std::shared_ptr<BridgeBiddingState> &state, const TensorDict &obs, const TensorDict &reply) {
    std::vector<int> bid_history = state->BidHistory();
    // Avoid zero
    int length = static_cast<int>(bid_history.size()) + 1;
    Player current_player = state->CurrentPlayer();
    // Get top actions
    auto probs = reply.at("raw_probs") * obs.at("legal_actions");
    probs /= torch::sum(probs);
    auto [top_k_actions, top_k_probs] = GetTopKActions(probs, params_.top_k, params_.min_prob);
    top_k_probs = top_k_probs.to(torch::kFloat64);
    int num_actions = static_cast<int>(top_k_actions.numel());
    std::vector<std::string> action_strings(num_actions);
    for (int i = 0; i < num_actions; ++i) {
      action_strings[i] = BidString(top_k_actions[i].item<Action>());
    }
    if (params_.verbose) {
      std::cout << "Get top k actions:\n";
      for (int i = 0; i < num_actions; ++i) {
        std::cout << action_strings[i] << ":" << top_k_probs[i].item<double>() << std::endl;
      }
    }
    // For a large min_prob
    if (num_actions == 1 || num_actions == 0) {
      return {
          {"policy_posterior", probs},
          {"a", probs.argmax()}
      };
    }

    if (top_k_probs[0].item<float>() >= params_.max_prob) {
      return {
          {"policy_posterior", probs},
          {"a", probs.argmax()}
      };
    }
    std::vector<Cards> sampled_cards;

    int num_rollouts = 0, num_particles = 0;
    torch::Tensor belief = belief_model_->GetBelief(obs);
    if (params_.verbose) {
      std::cout << "Get belief." << std::endl;
    }

    // Sample
    std::vector<int> remained_indices;
    auto vec_state = std::make_shared<VectorState>();
    std::vector<Cards> cards_vec;
    while (num_rollouts < params_.max_rollouts && num_particles < params_.max_particles) {
      Cards cards;
      if (params_.random_sample) {
        cards = RandomSample(state, current_player);
      } else {
        cards = Sample(belief, obs, current_player);
      }
      num_particles++;
      if (!cards.empty()) {
        cards_vec.push_back(cards);
      }
      if (cards_vec.size() == params_.filter_batch_size || num_particles == params_.max_particles) {
        remained_indices = FilterParticles(cards_vec, bid_history, current_player);
//        if (params_.verbose){
//          rl::utils::PrintVector(remained_indices);
//        }
        for (const auto idx : remained_indices) {
          const BridgeDeal deal{cards_vec[idx]};
          auto particle = std::make_shared<BridgeBiddingState>(deal);
          vec_state->Push(particle);
        }
        cards_vec.clear();
        num_rollouts += static_cast<int>(remained_indices.size());
      }
    }
    if (params_.verbose) {
      std::cout << "Sample and filter particles." << std::endl;
    }



    // Pop if too many
    if (vec_state->Size() > params_.max_rollouts) {
      vec_state->Pop(vec_state->Size() - params_.max_rollouts);
      num_rollouts = params_.max_rollouts;
    }
    if (params_.verbose) {
      std::cout << "Number of particle: " << num_particles << std::endl;
      std::cout << "Number of rollouts: " << num_rollouts << std::endl;
      std::cout << "Vec state size: " << vec_state->Size() << std::endl;
    }

    if (num_rollouts <= params_.min_rollouts) {
      return {
          {"policy_posterior", probs},
          {"a", probs.argmax()}
      };
    }

    // Build deals
    for (const auto &bid : bid_history) {
      vec_state->Step(bid);
    }
    if (params_.verbose) {
      std::cout << "Build deals." << std::endl;
//      vec_state->Show(5);
    }


    std::vector<std::vector<int>> indices_per_particle(num_rollouts);
    // Rollout
    torch::Tensor values = torch::zeros(num_actions, {torch::kFloat64});
    for (int i = 0; i < num_actions; ++i) {
      if (params_.verbose) {
        std::cout << "Rollout action: " << action_strings[i] << std::endl;
      }
      // size : num_rollouts
      auto contract_indices = Rollout(vec_state, actor_, top_k_actions[i].item<Action>(), current_player);
//      if(params_.verbose){
//        std::cout << "Indices for action " << action_strings[i] << std::endl;
//        rl::utils::PrintVector(contract_indices);
//      }
      // Push into particle indices
      for (size_t j = 0; j < num_rollouts; ++j) {
        indices_per_particle[j].push_back(contract_indices[j]);
      }
    }
    const std::vector<std::vector<double>> returns_per_particle =
        vec_state->ReturnsByContractIndices(indices_per_particle, current_player);
    for (const auto& scores : returns_per_particle) {
      for (int i = 0; i < num_actions; ++i) {
        values[i] += scores[i];
      }
    }
//    values[i] = utils::SumUpVector(contract_indices);
    if (params_.verbose) {
      std::cout << "Rollout values:\n";
      for (int i = 0; i < num_actions; ++i) {
        std::cout << action_strings[i] << ":" << values[i].item<double>() << std::endl;
      }
    }
    // Get policy
    torch::Tensor
        altered_probs = torch::pow(top_k_probs, params_.prob_exponent)
        * torch::exp(values * pow(length, params_.length_exponent) / (params_.temperature * sqrt(num_rollouts)));
    torch::Tensor policy_posterior = torch::zeros(kNumCalls, {torch::kFloat64});
    policy_posterior.scatter_(0, top_k_actions.to(torch::kLong), altered_probs);
    policy_posterior /= torch::sum(policy_posterior);
    auto action = policy_posterior.argmax();
    if (params_.verbose) {
      std::cout << "Policy Posterior:" << std::endl;
      rl::utils::PrintTensor<double>(policy_posterior);
    }
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
