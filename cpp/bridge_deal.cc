//
// Created by qzz on 2023/5/3.
//
#include "bridge_deal.h"
namespace rl::bridge {
BridgeDealManager::BridgeDealManager(const std::vector<Cards> &cards_vector,
                                     const std::vector<DDT> &ddts,
                                     const std::vector<int> &par_scores,
                                     int max_size)
    : cards_vector_(cards_vector),
      ddts_(ddts),
      par_scores_(par_scores),
      max_size_(max_size) {
  RL_CHECK_EQ(cards_vector_.size(), ddts_.size());
  RL_CHECK_EQ(cards_vector_.size(), par_scores_.size());
  size_ = static_cast<int>(cards_vector_.size());
}

BridgeDealManager::BridgeDealManager(const std::vector<Cards> &cards_vector,
                                     const std::vector<DDT> &ddts,
                                     const std::vector<int> &par_scores)
    : cards_vector_(cards_vector),
      ddts_(ddts),
      par_scores_(par_scores) {
  RL_CHECK_EQ(cards_vector_.size(), ddts_.size());
  RL_CHECK_EQ(cards_vector_.size(), par_scores_.size());
  size_ = static_cast<int>(cards_vector_.size());
  max_size_ = size_;
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

void BridgeDealManager::Add(const std::vector<Action> &cards, const std::vector<int> &ddt, int par_score) {
  std::lock_guard<std::mutex> lk(m_);
  if (size_ < max_size_) {
    cards_vector_.push_back(cards);
    ddts_.push_back(ddt);
    par_scores_.push_back(par_score);
    size_ += 1;
  } else {
    cards_vector_[add_cursor_] = cards;
    ddts_[add_cursor_] = ddt;
    par_scores_[add_cursor_] = par_score;
    add_cursor_ = (add_cursor_ + 1) % max_size_;
  }
}

}