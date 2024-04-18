//
// Created by qzz on 2023/6/28.
//
#include "search.h"
namespace rl::bridge {
const Cards all_cards = rl::utils::Arange(0, kNumCards);
bool CheckDealLegality(const Cards &cards) {
  Cards cards_copy = cards;
  std::sort(cards_copy.begin(), cards_copy.end());
  for (int i = 0; i < kNumCards; ++i) {
    if (cards_copy[i] != all_cards[i]) {
      return false;
    }
  }
  return true;
}

torch::Tensor BeliefModel::GetBelief(const TensorDict &obs) {
  std::vector<std::vector<Action>> cards_vector;
  torch::NoGradGuard ng;
  TorchJitInput input;
  int id = -1;
  auto model = model_locker_->GetModel(&id);
  auto s = obs.at("s").unsqueeze(0).to(model_locker_->device_);
  input.emplace_back(s);
  auto output = model.get_method("forward")(input);
  auto pred = output.toTensor().squeeze().cpu();
//  std::cout << pred << std::endl;
//  std::vector<float> pred_vec(pred.data_ptr<float>(), pred.data_ptr<float>() + pred.numel());
//  int i_sample = 0, i_try = 0;
//  while (i_sample < num_sample && i_try < max_try) {
//    auto sample_ret = Sample_(pred, obs, current_player);
//    i_try++;
//    const bool is_legal = CheckDealLegality(sample_ret);
//    if (is_legal) {
//      cards_vector.push_back(sample_ret);
//      i_sample += 1;
//    }
////    rl::utils::PrintVector(sample_ret);
//  }
  return pred;
}

Cards Sample(const torch::Tensor &belief_pred, const TensorDict &obs, Player current_player) {
  const torch::Tensor basic_indices = torch::tensor({0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48}, {torch::kInt64});

  const auto player_cards_feature = obs.at("s").narrow(0, kObservationTensorSize - kNumCards, kNumCards);
  const auto player_cards = torch::nonzero(player_cards_feature).squeeze().to(torch::kInt32);
  // Cards have been selected
  torch::Tensor deal_cards = torch::ones(kNumCards, {torch::kInt32}).fill_(-1);
  deal_cards.scatter_(0, basic_indices + current_player, player_cards);
  torch::Tensor selected_cards = player_cards_feature.clone();
//  std::cout << player_cards_feature << std::endl;
  for (const int sample_relative_player : {2, 1, 3}) {
    const int start_index = (sample_relative_player - 1) * kNumCards;
    const int end_index = sample_relative_player * kNumCards;
    torch::Tensor relative_player_pred = belief_pred.slice(0, start_index, end_index).clone();
    relative_player_pred *= 1 - selected_cards;
    if (torch::count_nonzero(relative_player_pred).item<int>() < kNumCardsPerHand) {
      return {}; // sentinel
    }
    // Sample 13 cards
    torch::Tensor sample_cards = torch::multinomial(relative_player_pred, kNumCardsPerHand, false);
    selected_cards.scatter_(0, sample_cards, 1);
    deal_cards.scatter_(0,
                        basic_indices + (current_player + sample_relative_player) % kNumPlayers,
                        sample_cards.to(torch::kInt32));
  }
  std::vector<int> ret(deal_cards.data_ptr<int>(), deal_cards.data_ptr<int>() + deal_cards.numel());
  bool is_legal = CheckDealLegality(ret);
  return is_legal ? ret : std::vector<int>{};
}

TensorDict VectorState::GetFeature() const {
  std::vector<TensorDict> obs_vec;
  obs_vec.reserve(states_.size());
  for (const auto &state : states_) {
    obs_vec.push_back(MakeObsTensorDict(state, 1));
  }
  return tensor_dict::Stack(obs_vec, 0);
}
void VectorState::Step(const TensorDict &reply) {
  for (size_t i = 0; i < states_.size(); ++i) {
    if (!states_[i]->Terminated()) {
      auto rep = tensor_dict::Index(reply, i);
      states_[i]->ApplyAction(rep.at("a").item<Action>());
    }
  }
}
bool VectorState::AllTerminated() const {
  for (const auto &state : states_) {
    if (!state->Terminated()) {
      return false;
    }
  }
  return true;
}
void VectorState::Step(const Action bid) {
  for (size_t i = 0; i < states_.size(); ++i) {
    states_[i]->ApplyAction(bid);
  }
}
std::shared_ptr<VectorState> VectorState::Clone() const {
  auto cloned = std::make_shared<VectorState>();
  for (size_t i = 0; i < states_.size(); ++i) {
    auto state = states_[i]->Clone();
    cloned->Push(state);
  }
  return cloned;
}
std::vector<double> VectorState::GetReturns(Player player) const {
  RL_CHECK_TRUE(AllTerminated())
  std::vector<double> returns;
  for (size_t i = 0; i < states_.size(); ++i) {
    returns.push_back(states_[i]->Returns()[player]);
  }
  return returns;
}
void VectorState::Pop(int num_pop) {
  RL_CHECK_GE(num_pop, 0);
  int real_num_pop = std::min(Size(), num_pop);
  for (int i = 0; i < real_num_pop; ++i) {
    states_.pop_back();
  }
}

TensorDict VectorState::GetFinalObservation() const {
  RL_CHECK_TRUE(AllTerminated())
  std::vector<int> mask(Size());
  std::vector<torch::Tensor> final_obs_vec;
  torch::Tensor final_obs;
  for (size_t i = 0; i < states_.size(); ++i) {
    auto contract = states_[i]->GetContract();
    if (contract.level > 0) {
      mask[i] = 1;
      final_obs = torch::tensor(states_[i]->FinalObservationTensor());
    } else {
      mask[i] = 0;
      final_obs = torch::zeros(kFinalTensorSize, {torch::kFloat});
    };
    final_obs_vec.push_back(final_obs);
  }
  TensorDict ret = {
      {"final_s", torch::stack(final_obs_vec, 0)},
      {"not_passed_out_mask", torch::tensor(mask)}
  };
  return ret;
}
std::vector<int> VectorState::ContractIndices() const {
  RL_CHECK_TRUE(AllTerminated())
  std::vector<int> contract_indices(states_.size());
  for (size_t i = 0; i < states_.size(); ++i) {
    int index = states_[i]->ContractIndex();
    contract_indices[i] = index;
  }
  return contract_indices;
}
std::vector<std::vector<double>> VectorState::ReturnsByContractIndices(const std::vector<std::vector<int>>& indices,
                                                                       const Player player) const {
  std::vector<std::vector<double>> ret(states_.size());
  for (size_t i = 0; i < states_.size(); ++i) {
    const std::vector<int> scores = states_[i]->ScoreForContracts(player, indices[i]);
    const std::vector<double> double_scores(scores.begin(), scores.end());
    ret[i] = double_scores;
  }
  return ret;
}

std::vector<int> Rollout(const std::shared_ptr<VectorState> &vec_state,
                         const std::shared_ptr<VecEnvActor> &actor,
                         Action bid, Player player) {
  auto cloned = vec_state->Clone();
  cloned->Step(bid);
  TensorDict obs, reply;
  while (!cloned->AllTerminated()) {
    obs = cloned->GetFeature();
    reply = actor->Act(obs);
    cloned->Step(reply);
  }
  return cloned->ContractIndices();
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
Cards RandomSample(const std::shared_ptr<BridgeBiddingState> &state, Player player) {
  const std::vector<Action> player_cards = state->GetPlayerCards(player);
  std::vector<Action> remained_cards;
  std::set_difference(all_cards.begin(), all_cards.end(),
                      player_cards.begin(), player_cards.end(),
                      std::back_inserter(remained_cards));
  std::shuffle(remained_cards.begin(), remained_cards.end(), std::random_device());
  for (int i = 0; i < player_cards.size(); ++i) {
    remained_cards.insert(remained_cards.begin() + i * kNumPlayers + player, player_cards[i]);
  }
  return remained_cards;
}

std::vector<double> ScorePredictor::Predict(const std::shared_ptr<VectorState> &vec_state, Player player) {
  auto final_obs = vec_state->GetFinalObservation();
  final_obs.insert({"player", torch::tensor(player)});
  torch::NoGradGuard ng;
  TorchJitInput input;
  input.emplace_back(rl::tensor_dict::ToTorchDict(final_obs, model_locker_->device_));
  int id = -1;
  auto model = model_locker_->GetModel(&id);
  auto output = model.get_method("predict")(input);
  auto reply = rl::tensor_dict::FromIValue(output, torch::kCPU, true);
  torch::Tensor scores_tensor = reply.at("scores");
  std::vector<double> scores;
  scores.reserve(scores_tensor.numel());
  for (int i = 0; i < scores_tensor.numel(); ++i) {
    scores.push_back(scores_tensor[i].item<float>());
  }
  return scores;
}
}