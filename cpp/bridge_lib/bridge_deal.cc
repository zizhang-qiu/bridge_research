//
// Created by qzz on 2023/5/3.
//
#include "bridge_deal.h"
namespace rl::bridge {

BridgeDealManager::BridgeDealManager(const std::vector<Cards> &cards_vector) {
  RL_CHECK_GT(cards_vector.size(), 0);
  MakeDealsFromCards(cards_vector);
  size_ = static_cast<int>(cards_vector.size());
}

BridgeDealManager::BridgeDealManager(const std::vector<Cards> &cards_vector,
                                     const std::vector<DDT> &ddts) {
  RL_CHECK_GT(cards_vector.size(), 0);
  RL_CHECK_EQ(cards_vector.size(), ddts.size());
  MakeDealsFromCardsAndDDT(cards_vector, ddts);
  size_ = static_cast<int>(cards_vector.size());

}

BridgeDeal BridgeDealManager::Next() {
  std::lock_guard<std::mutex> lk(m_);
  RL_CHECK_GT(size_, 0);
  BridgeDeal deal = deals_[cursor_];
  cursor_ = (cursor_ + 1) % size_;
  return deal;
}

void BridgeDealManager::Reset() {
  std::lock_guard<std::mutex> lk(m_);
  cursor_ = 0;
}

void BridgeDealManager::Update(const std::vector<Cards> &cards_vector,
                               const std::vector<DDT> &ddts) {
  std::lock_guard<std::mutex> lk(m_);
  RL_CHECK_EQ(cards_vector.size(), ddts.size());
  RL_CHECK_GT(cards_vector.size(), 0);
  deals_.clear();
  size_ = static_cast<int>(cards_vector.size());
  cursor_ = 0;
  MakeDealsFromCardsAndDDT(cards_vector, ddts);
}
void BridgeDealManager::Update(const std::vector<Cards> &cards_vector) {
  RL_CHECK_GT(cards_vector.size(), 0);
  deals_.clear();
  MakeDealsFromCards(cards_vector);
  size_ = static_cast<int>(cards_vector.size());
  cursor_ = 0;
}
void BridgeDealManager::MakeDealsFromCards(const std::vector<Cards> &cards_vector) {
  for (const auto &cards : cards_vector) {
    const BridgeDeal deal{cards, kNorth, false, false};
    deals_.push_back(deal);
  }
}
void BridgeDealManager::MakeDealsFromCardsAndDDT(const std::vector<Cards> &cards_vector, const std::vector<DDT> &ddts) {
  for (int i = 0; i < cards_vector.size(); ++i) {
    const BridgeDeal deal{cards_vector[i], kNorth, false, false, ddts[i]};
    deals_.push_back(deal);
  }
}

}