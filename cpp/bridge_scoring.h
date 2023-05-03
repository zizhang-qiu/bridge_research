//
// Created by qzz on 2023/2/23.
//

#ifndef BRIDGE_RESEARCH_BRIDGE_SCORING_H
#define BRIDGE_RESEARCH_BRIDGE_SCORING_H
#include <array>
#include <string>
#include "rl/utils.h"
#include "rl/logging.h"
#include "str_utils.h"
#include "bridge_constants.h"
namespace rl::bridge {

struct Contract {
  int level = 0;
  Denomination trumps = kNoTrump;
  DoubleStatus double_status = kUndoubled;
  int declarer = -1;

  std::string ToString() const;
  int Index() const;
};

int ComputeScore(Contract contract, int declarer_tricks, bool is_vulnerable);

// All possible contracts.
inline constexpr int kNumContracts =
    kNumBids * kNumPlayers * kNumDoubleStates + 1;

constexpr std::array<Contract, kNumContracts> AllContracts() {
  std::array<Contract, kNumContracts> contracts;
  int i = 0;
  contracts[i++] = Contract();
  for (int level : {1, 2, 3, 4, 5, 6, 7}) {
    for (Denomination trumps :
        {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
        for (DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          contracts[i++] = Contract{level, trumps, double_status, declarer};
        }
      }
    }
  }
  return contracts;
}

inline constexpr std::array<Contract, kNumContracts> kAllContracts =
    AllContracts();

std::vector<int> AllScores();

int GetImp(int score1, int score2);
}
#endif //BRIDGE_RESEARCH_BRIDGE_SCORING_H
