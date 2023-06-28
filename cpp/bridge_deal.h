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
  std::optional<int> par_score;
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
  BridgeDealManager(const std::vector<Cards> &cards_vector,
                    const std::vector<DDT> &ddts,
                    const std::vector<int> &par_scores,
                    int max_size);

  BridgeDealManager(const std::vector<Cards> &cards_vector,
                    const std::vector<DDT> &ddts,
                    const std::vector<int> &par_scores);

  BridgeDeal Next() override;
  void Reset();
  void Add(const std::vector<Action> &cards, const std::vector<int> &ddt, int par_score);
  int Size() const { return size_; };
  int Cursor() const { return cursor_.load(); }
  std::tuple<Cards, DDT, int> Get(int index) {
    return std::make_tuple(cards_vector_[index], ddts_[index], par_scores_[index]);
  }

 private:
  std::mutex m_;
  std::atomic<int> cursor_ = 0;
  std::atomic<int> add_cursor_ = 0;
  int size_;
  int max_size_;
  std::vector<Cards> cards_vector_;
  std::vector<DDT> ddts_;
  std::vector<int> par_scores_;
};

class RandomDealManager: public DealManager {
 public:
  RandomDealManager(int seed) : rng_(seed) {}

  BridgeDeal Next() override{
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
