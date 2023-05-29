//
// Created by qzz on 2023/5/23.
//

#ifndef BRIDGE_RESEARCH_CPP_SAYC_BOT_H_
#define BRIDGE_RESEARCH_CPP_SAYC_BOT_H_
#include "bridge_state.h"
#include "rl/utils.h"
namespace rl::bridge {
inline constexpr int kNoAction = -1;
class SAYCBot {
 public:
  Action NoTrumpAction(int hcp);

  std::vector<Action> OpeningLegalActions(const std::shared_ptr<BridgeBiddingState> &state);

  bool CheckBalancedHand(HandEvaluation hand_evaluation);

  bool CheckRuleOfTwenty(HandEvaluation hand_evaluation);

  Denomination OneLevelOpeningSuit(HandEvaluation hand_evaluation);

  SAYCBot() = default;
};

}
#endif //BRIDGE_RESEARCH_CPP_SAYC_BOT_H_
