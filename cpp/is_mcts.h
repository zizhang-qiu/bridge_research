//
// Created by qzz on 2023/6/18.
//

#ifndef BRIDGE_RESEARCH_CPP_IS_MCTS_H_
#define BRIDGE_RESEARCH_CPP_IS_MCTS_H_
#include "bridge_lib/bridge_state.h"
namespace rl::bridge{
using ActionsAndProbs = std::vector<std::pair<Action, double>>;

// Abstract class representing an evaluation function for a game.
// The evaluation function takes in an intermediate state in the game and
// returns an evaluation of that state, which should correlate with chances of
// winning the game for player 0.
class Evaluator {
 public:
  virtual ~Evaluator() = default;

  // Return a value of this state for each player.
  virtual std::vector<double> Evaluate(const State& state) = 0;

  // Return a policy: the probability of the current player playing each action.
  virtual ActionsAndProbs Prior(const State& state) = 0;
};

class NetworkBasedEvaluator : public Evaluator{

};

// Use this constant to use an unlimited number of world samples.
inline constexpr int kUnlimitedNumWorldSamples = -1;

// The key identifying a node contains the InformationStateString or
// ObservationString, as well as the player id, because in some games the
// observation string can be the same for different players.
using ISMCTSStateKey = std::pair<Player, std::string>;

enum class ISMCTSFinalPolicyType {
  kNormalizedVisitCount,
  kMaxVisitCount,
  kMaxValue,
};

struct ChildInfo {
  int visits;
  double return_sum;
  double value() const { return return_sum / visits; }
};

struct ISMCTSNode {
  std::unordered_map<Action, ChildInfo> child_info;
  int total_visits;
};

using InfostateResampler = std::function<std::unique_ptr<State>(
    const State& state, Player pl, std::function<double()> rng)>;

}
#endif //BRIDGE_RESEARCH_CPP_IS_MCTS_H_
