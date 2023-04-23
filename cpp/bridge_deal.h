//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
#include "bridge_scoring.h"
#include "types.h"

namespace rl::bridge{
enum Seat { kNorth, kEast, kSouth, kWest };
struct BridgeDeal {
  Cards cards;
  Player dealer = kNorth;
  bool is_dealer_vulnerable = false;
  bool is_non_dealer_vulnerable = false;
  std::optional<DDT> ddt;
  std::optional<int> par_score;
};

// A manager stores deals
class BridgeDealManager {
public:
  BridgeDealManager(const std::vector<Cards> &cards_vector,
                    const std::vector<DDT> &ddts,
                    const std::vector<int> &par_scores) : cards_vector_(cards_vector),
                                                          ddts_(ddts), par_scores_(par_scores) {
    RL_CHECK_EQ(cards_vector_.size(), ddts_.size());
    RL_CHECK_EQ(cards_vector_.size(), par_scores_.size());
    size_ = static_cast<int>(cards_vector_.size());
  };

  BridgeDeal Next() {
    std::lock_guard<std::mutex> lk(m_);
    BridgeDeal deal{cards_vector_[cursor_], kNorth, false, false, ddts_[cursor_], par_scores_[cursor_]};
    cursor_ = (cursor_ + 1) % size_;
    return deal;
  }

  void Reset() {
    std::lock_guard<std::mutex> lk(m_);
    cursor_ = 0;
  }

  int Size() const {
    return size_;
  }

private:
  std::mutex m_;
  int cursor_ = 0;
  int size_;
  std::vector<Cards> cards_vector_;
  std::vector<DDT> ddts_;
  std::vector<int> par_scores_;
};
}
#endif // BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
