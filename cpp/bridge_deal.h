//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
#include "bridge_constants.h"
#include "rl/logging.h"

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

// A manager stores deals
class BridgeDealManager {
 public:
  BridgeDealManager(const std::vector<Cards> &cards_vector,
                    const std::vector<DDT> &ddts,
                    const std::vector<int> &par_scores);

  BridgeDeal Next();
  void Reset();
  int Size() const { return size_;};
  int Cursor() const { return cursor_.load(); }

 private:
  std::mutex m_;
  std::atomic<int> cursor_ = 0;
  int size_;
  std::vector<Cards> cards_vector_;
  std::vector<DDT> ddts_;
  std::vector<int> par_scores_;
};
}
#endif // BRIDGE_RESEARCH_CPP_BRIDGE_DEAL_H_
