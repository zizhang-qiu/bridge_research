//
// Created by qzz on 2023/5/3.
//
#include "bridge_deal.h"
namespace rl::bridge {
BridgeDealManager::BridgeDealManager(const std::vector<Cards> &cards_vector,
                                     const std::vector<DDT> &ddts,
                                     const std::vector<int> &par_scores)
    : cards_vector_(cards_vector),
      ddts_(ddts), par_scores_(par_scores) {
  RL_CHECK_EQ(cards_vector_.size(), ddts_.size());
  RL_CHECK_EQ(cards_vector_.size(), par_scores_.size());
  size_ = static_cast<int>(cards_vector_.size());
}

BridgeDeal BridgeDealManager::Next() {
  std::lock_guard<std::mutex> lk(m_);
  BridgeDeal deal{cards_vector_[cursor_], kNorth, false, false, ddts_[cursor_], par_scores_[cursor_]};
  cursor_ = (cursor_ + 1) % size_;
  return deal;
}

void BridgeDealManager::Reset() {
  std::lock_guard<std::mutex> lk(m_);
  cursor_ = 0;
}

}