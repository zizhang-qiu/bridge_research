//
// Created by qzz on 2023/7/7.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_MOVE_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_MOVE_H_
#include "bridge_constants.h"
namespace rl::bridge {
class BridgeMove {
 public:
  enum Type { kInvalid, kDeal, kAuction, kPlay };
  BridgeMove(Type move_type, int move_index) : move_type_(move_type), move_index_(move_index) {}
 private:
  Type move_type_ = kInvalid;
  int move_index_;
  int bid_level = -1;
  Denomination bid_trump = kNoDenomination;
  int card_rank = -1;
  Suit suit = Suit::kClubs;
};
}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_MOVE_H_
