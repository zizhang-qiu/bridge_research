//
// Created by qzz on 2023/7/7.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_GAME_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_GAME_H_
#include <utility>

#include "game_parameters.h"
#include "bridge_constants.h"
#include "bridge_scoring.h"

namespace rl::bridge {
inline constexpr bool kDefaultVulnerability = false;
class BridgeBiddingGame{
  explicit BridgeBiddingGame(const Params &params) {
    params_ = params;
    is_dealer_vulnerable = ParameterValue<bool>(params_, "is_dealer_vulnerable", false);
    is_non_dealer_vulnerable = ParameterValue<bool>(params_, "is_non_dealer_vulnerable", false);
  }

  int NumDistinctActions() const {
    return kNumCards + kNumCalls;
  }

  int MaxChanceOutcomes() const { return kNumCards; }

  int NumPlayers() const{return kNumPlayers;}

  double MinUtility() const{return -kMaxScore;}

  double MaxUtility() const{return kMaxScore;}

  double UtilitySum() const{return 0;}

  std::string Name() const{return "BridgeBidding";}

  int MaxGameLength() const{return kNumCards + kMaxAuctionLength;}

  int NumPossibleContract() const{return kNumContracts;}

  bool IsDealerVulnerable() const{return is_dealer_vulnerable;}

  bool IsNonDealerVulnerable() const{return is_non_dealer_vulnerable;}

 private:
  Params params_;
  bool is_dealer_vulnerable;
  bool is_non_dealer_vulnerable;
};
}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_GAME_H_
