//
// Created by qzz on 2023/5/25.
//
#include "sayc_bot.h"
namespace rl::bridge {

bool SAYCBot::CheckBalancedHand(const HandEvaluation hand_evaluation) {
  auto length_per_suit = hand_evaluation.length_per_suit;
  int counts[14] = {0};
  for (const Suit suit : {Suit::kClubs, Suit::kDiamonds, Suit::kSpades, Suit::kHearts}) {
    int this_suit_length = length_per_suit[static_cast<int>(suit)];
    // In a balanced hand, there is no singleton or void
    counts[this_suit_length]++;
  }

  // a balanced hands is 4-3-3-3, 4-4-3-2 or 5-3-3-2
  if ((counts[4] == 1 && counts[3] == 3)
      || (counts[4] == 2 && counts[3] == 1 && counts[2] == 1)
      || (counts[5] == 1 && counts[3] == 2 && counts[2] == 1)) {
    return true;
  }
  return false;
}

bool SAYCBot::CheckRuleOfTwenty(const HandEvaluation hand_evaluation) {
  auto two_longest = utils::GetTopKElements(hand_evaluation.length_per_suit, 2);
  int point = utils::SumUpVector(two_longest) + hand_evaluation.high_card_points;
  return point >= 20;
}

Action SAYCBot::NoTrumpAction(int hcp) {
  if (15 <= hcp && hcp <= 17) {
    return Bid(1, kNoTrump);
  }
  if (20 <= hcp && hcp <= 21) {
    return Bid(2, kNoTrump);
  }
  if (25 <= hcp && hcp <= 27) {
    return Bid(3, kNoTrump);
  }
  return kNoAction;
}
std::vector<Action> SAYCBot::OpeningLegalActions(const std::shared_ptr<BridgeBiddingState> &state) {
  std::vector<Action> legal_actions = {kPass};
  Player current_player = state->CurrentPlayer();
  auto player_cards = state->GetPlayerCards(current_player);
  auto player_hand_evaluation = state->GetHandEvaluation()[current_player];
  // check balanced and no trump
  bool is_balanced = CheckBalancedHand(player_hand_evaluation);
  if (is_balanced) {
    Action no_trump_action = NoTrumpAction(player_hand_evaluation.high_card_points);
    if (no_trump_action != kNoAction) {
      return {no_trump_action};
    }
  }

  bool FitRuleOfTwenty = CheckRuleOfTwenty(player_hand_evaluation);
  if (FitRuleOfTwenty){

  }

  auto length_per_suit = player_hand_evaluation.length_per_suit;
  // 1d with 4-4 in minors
  if (length_per_suit[static_cast<int>(Suit::kClubs)] == 4
      && length_per_suit[static_cast<int>(Suit::kDiamonds)] == 4) {
    return {Bid(1, kDiamonds)};
  }
  // 1c with 3-3 in minors
  if (length_per_suit[static_cast<int>(Suit::kClubs)] == 3
      && length_per_suit[static_cast<int>(Suit::kDiamonds)] == 3) {
    return {Bid(1, kClubs)};
  }

  return legal_actions;
}
Denomination SAYCBot::OneLevelOpeningSuit(const HandEvaluation hand_evaluation) {
  return kClubs;
}
}