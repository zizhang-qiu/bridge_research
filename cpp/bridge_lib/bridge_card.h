//
// Created by qzz on 2023/7/7.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_CARD_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_CARD_H_
#include "bridge_constants.h"
namespace rl::bridge {
class BridgeCard {
 public:
  BridgeCard(Suit suit, int rank);

  BridgeCard(int index);

  std::string ToString() const;

  int Index() const;

  int Rank() const{return rank_;}
 private:
  Suit suit_;
  int rank_;
};


}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_CARD_H_
