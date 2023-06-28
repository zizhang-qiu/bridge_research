//
// Created by qzz on 2023/5/21.
//
#include "dds.h"
namespace rl::bridge {
int trump_filter[5] = {0, 0, 0, 0, 0};

std::vector<Player> GetHolder(const std::vector<Action> &cards) {
  std::vector<Player> holder(kNumCards);
  for (int i = 0; i < kNumCards; ++i) {
    auto card = cards[i];
    Player card_holder = i % kNumPlayers;
    holder[card] = card_holder;
  }
  return holder;
}

ddTableDeal Holder2ddTableDeal(const std::vector<Player> &holder) {
  ddTableDeal deal{};
  for (const auto suit : {Suit::kClubs, Suit::kDiamonds, Suit::kHearts, Suit::kSpades}) {
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      Player player = holder[static_cast<int>(suit) + rank * kNumSuits];
      deal.cards[player][SuitToDDSStrain(suit)] += 1 << (2 + rank);
    }
  }
  return deal;
}

ddTablesRes CalcBatchDDTs(const std::vector<std::vector<Action>> &cards_vector, int mode) {
  int batch_size = static_cast<int>(cards_vector.size());
//  std::cout << batch_size << std::endl;
  RL_CHECK_LE(batch_size, kMaxDDSBatchSize);
  ddTableDeals dealsp{};
  dealsp.noOfTables = batch_size;
  ddTablesRes res{};
  for (int i = 0; i < batch_size; ++i) {
    auto holder = GetHolder(cards_vector[i]);
    ddTableDeal tabel = Holder2ddTableDeal(holder);
    dealsp.deals[i] = tabel;
  }
  allParResults pres{};

  const int return_code = CalcAllTables(&dealsp, mode, trump_filter, &res, &pres);
  if (return_code != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(return_code, error_message);
    std::cerr << utils::StrCat("double_dummy_solver:", error_message)
              << std::endl;
  }
  return res;
}

std::vector<ddTableResults> CalcDDTs(const std::vector<std::vector<Action>> &cards_vector, const int mode, int num_threads) {
  int num_deals = static_cast<int>(cards_vector.size());
  int num_batches = ceil(static_cast<float>(num_deals) / static_cast<float>(kMaxDDSBatchSize));
  std::vector<ddTableResults> results;
  SetMaxThreads(num_threads);
  int left, right;
  for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    left = batch_idx * kMaxDDSBatchSize;
    right = std::min(left + kMaxDDSBatchSize, num_deals);
    int num_deals_this_batch = right - left;
    std::vector<std::vector<Action>> batch_cards_vector(cards_vector.begin() + left, cards_vector.begin() + right);
    auto table_res = CalcBatchDDTs(batch_cards_vector, mode);
    auto table_results = table_res.results;
    for (int i = 0; i < num_deals_this_batch; ++i) {
      results.emplace_back(table_results[i]);
    }
  }
  RL_CHECK_EQ(results.size(), num_deals);
  return results;
}

ddTableResults CalcOneDeal(const std::vector<Action> &cards) {
  auto holder = GetHolder(cards);
  ddTableDeal deal = Holder2ddTableDeal(holder);
  ddTableResults double_dummy_results{};
  SetMaxThreads(1);
  const int return_code = CalcDDtable(deal, &double_dummy_results);
  if (return_code != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(return_code, error_message);
    std::cerr << utils::StrCat("double_dummy_solver:", error_message)
              << std::endl;
  }
  return double_dummy_results;
}

std::vector<int> ddTableResults2ddt(const ddTableResults double_dummy_results) {
  auto res_table = double_dummy_results.resTable;
  std::vector<int> ddt(kDoubleDummyResultSize);
  for (auto denomination : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
    for (auto player : {kNorth, kEast, kSouth, kWest}) {
      ddt[denomination * kNumPlayers + player] =
          res_table[DenominationToDDSStrain(denomination)][player];
    }
  }
  return ddt;
}

std::tuple<std::vector<Action>, std::vector<int>> GenerateOneDeal(std::mt19937 &rng) {
  std::vector<Action> cards = utils::Permutation(0, kNumCards, rng);
  auto holder = GetHolder(cards);
  ddTableResults double_dummy_results = CalcOneDeal(cards);
  auto ddt = ddTableResults2ddt(double_dummy_results);
  return std::make_tuple(cards, ddt);
}



}