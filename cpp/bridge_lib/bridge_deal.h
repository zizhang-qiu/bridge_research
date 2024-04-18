//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
#include "bridge_constants.h"
#include "rl/logging.h"
#include "rl/utils.h"
#include <random>

namespace rl::bridge {
enum Seat { kNorth, kEast, kSouth, kWest };
struct BridgeDeal {
  Cards cards;
  Player dealer = kNorth;
  bool is_dealer_vulnerable = false;
  bool is_non_dealer_vulnerable = false;
  std::optional<DDT> ddt;
};

class DealManager {
 public:
  DealManager() = default;

  ~DealManager() = default;

  virtual BridgeDeal Next() = 0;
};

// A manager stores deals
class BridgeDealManager : public DealManager {
 public:
  BridgeDealManager() = default;

  explicit BridgeDealManager(const std::vector<Cards> &cards_vector);

  BridgeDealManager(const std::vector<Cards> &cards_vector, const std::vector<DDT> &ddts);

  BridgeDeal Next() override;
  void Reset();
  int Size() const { return size_; };
  void Update(const std::vector<Cards> &cards_vector,
              const std::vector<DDT> &ddts);

  void Update(const std::vector<Cards> &cards_vector);
  int Cursor() const { return cursor_.load(); }
//  std::tuple<Cards, DDT> Get(int index) const {
//    RL_CHECK_LT(index, size_);
//    return std::make_tuple(cards_vector_[index], ddts_[index]);
//  }

 private:
  std::mutex m_;
  std::atomic<int> cursor_ = 0;
  int size_{};
  std::vector<BridgeDeal> deals_;
  void MakeDealsFromCards(const std::vector<Cards> &cards_vector);
  void MakeDealsFromCardsAndDDT(const std::vector<Cards> &cards_vector, const std::vector<DDT> &ddts);
};

class RandomDealManager : public DealManager {
 public:
  explicit RandomDealManager(int seed) : rng_(seed) {}

  BridgeDeal Next() override {
    auto cards = rl::utils::Permutation(0, kNumCards, rng_);
    BridgeDeal deal;
    deal.cards = cards;
    deal.ddt = ddt_;
    return deal;
  }

 private:
  std::mt19937 rng_;
  const DDT ddt_ = utils::Zeros<int>(kDoubleDummyResultSize);
};
}
#endif // BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
