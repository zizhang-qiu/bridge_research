//
// Created by qzz on 2023/7/7.
//
#include "bridge_card.h"
std::string rl::bridge::BridgeCard::ToString() const {
  return {kSuitChar[static_cast<int>(suit_)], kRankChar[rank_]};
}
int rl::bridge::BridgeCard::Index() const {
  return rank_ * kNumSuits + static_cast<int>(suit_);
}
rl::bridge::BridgeCard::BridgeCard(rl::bridge::Suit suit, int rank) :
    suit_(suit), rank_(rank) {
  assert(rank_ >= 0 && rank < kNumCardsPerSuit);
}
rl::bridge::BridgeCard::BridgeCard(int index) {
  assert(index>=0 && index < kNumCards);
  suit_ = Suit(index % kNumSuits);
  rank_ = index / kNumSuits;
}
