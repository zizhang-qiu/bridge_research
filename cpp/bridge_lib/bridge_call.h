//
// Created by qzz on 2023/7/7.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_CALL_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_CALL_H_
#include "bridge_constants.h"
namespace rl::bridge {
class BridgeCall {
 public:
  BridgeCall() = default;

 private:
  Denomination denomination_ = kNoDenomination;
  int level = -1;

};
}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_LIB_BRIDGE_CALL_H_
