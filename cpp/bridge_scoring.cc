//
// Created by qzz on 2023/5/2.
//
#include "bridge_scoring.h"
namespace rl::bridge {
std::string Contract::ToString() const {
  if (level == 0) return "Passed Out";
  std::string str = utils::StrCat(level, std::string{kDenominationChar[trumps]});
  if (double_status == kDoubled) utils::StrAppend(&str, "X");
  if (double_status == kRedoubled) utils::StrAppend(&str, "XX");
  utils::StrAppend(&str, " ", std::string{kPlayerChar[declarer]});
  return str;
}

int Contract::Index() const {
  if (level == 0) return 0;
  int index = level - 1;
  index *= kNumDenominations;
  index += static_cast<int>(trumps);
  index *= kNumPlayers;
  index += static_cast<int>(declarer);
  index *= kNumDoubleStates;
  if (double_status == kRedoubled) index += 2;
  if (double_status == kDoubled) index += 1;
  return index + 1;
}

constexpr int kBaseTrickScores[] = {20, 20, 30, 30, 30};

int ComputeScoreContract(Contract contract, DoubleStatus double_status) {
  int score = contract.level * kBaseTrickScores[contract.trumps];
  if (contract.trumps == kNoTrump) score += 10;
  return score * double_status;
}

// ComputeScore for failing to make the contract (will be negative).
int ComputeScoreUndertricks(int undertricks, bool is_vulnerable,
                            DoubleStatus double_status) {
  if (double_status == kUndoubled) {
    return (is_vulnerable ? -100 : -50) * undertricks;
  }
  int score = 0;
  if (is_vulnerable) {
    score = -200 - 300 * (undertricks - 1);
  } else {
    if (undertricks == 1) {
      score = -100;
    } else if (undertricks == 2) {
      score = -300;
    } else {
      // This takes into account the -100 for the fourth and subsequent tricks.
      score = -500 - 300 * (undertricks - 3);
    }
  }
  return score * (double_status / 2);
}

// ComputeScore for tricks made in excess of the bid.
int ComputeScoreOvertricks(Denomination trump_suit, int overtricks, bool is_vulnerable,
                           DoubleStatus double_status) {
  if (double_status == kUndoubled) {
    return overtricks * kBaseTrickScores[trump_suit];
  } else {
    return (is_vulnerable ? 100 : 50) * overtricks * double_status;
  }
}

// Bonus for making a doubled or redoubled contract.
int ComputeScoreDoubledBonus(DoubleStatus double_status) {
  return 50 * (double_status / 2);
}

// Bonuses for partscore, game, or slam.
int ComputeScoreBonuses(int level, int contract_score, bool is_vulnerable) {
  if (level == 7) {  // 1500/1000 for grand slam + 500/300 for game
    return is_vulnerable ? 2000 : 1300;
  } else if (level == 6) {  // 750/500 for small slam + 500/300 for game
    return is_vulnerable ? 1250 : 800;
  } else if (contract_score >= 100) {  // game bonus
    return is_vulnerable ? 500 : 300;
  } else {  // partscore bonus
    return 50;
  }
}

int ComputeScore(Contract contract, int declarer_tricks, bool is_vulnerable) {
  if (contract.level == 0) return 0;
  int contracted_tricks = 6 + contract.level;
  int contract_result = declarer_tricks - contracted_tricks;
  if (contract_result < 0) {
    return ComputeScoreUndertricks(-contract_result, is_vulnerable,
                                   contract.double_status);
  } else {
    int contract_score = ComputeScoreContract(contract, contract.double_status);
    int bonuses = ComputeScoreBonuses(contract.level, contract_score, is_vulnerable) +
        ComputeScoreDoubledBonus(contract.double_status) +
        ComputeScoreOvertricks(contract.trumps, contract_result,
                               is_vulnerable, contract.double_status);
    return contract_score + bonuses;
  }
}

std::vector<int> AllScores() {
  std::vector<int> ret;

  for (int level = 0; level <= 7; ++level) {
    for (int tricks = 0; tricks <= kNumCardsPerSuit; tricks++) {
      for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
        for (const auto double_status : {kUndoubled, kDoubled, kRedoubled}) {
          for(bool vul : {true, false}) {
            const Contract contract{level, trump, double_status};
            int score = ComputeScore(contract, tricks, vul);
            ret.emplace_back(score);
            ret.emplace_back(-score);
          }
        }
      }
    }
  }
//  std::cout << ret.size() << std::endl;
  std::set<int> all_scores(ret.begin(), ret.end());
  std::vector<int> all_scores_vec(all_scores.begin(), all_scores.end());
  return all_scores_vec;
}

constexpr int kScoreTable[] = {15, 45, 85, 125, 165, 215, 265, 315,
                               365, 425, 495, 595, 745, 895, 1095, 1295,
                               1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995};
constexpr int kScoreTableSize = sizeof(kScoreTable) / sizeof(int);

int GetImp(int score1, int score2) {
  RL_CHECK_GE(score1, -kMaxScore);
  RL_CHECK_GE(score2, -kMaxScore);
  RL_CHECK_LE(score1, kMaxScore);
  RL_CHECK_LE(score2, kMaxScore);
  const int score = score1 - score2;
  const int sign = score == 0 ? 0 : (score > 0 ? 1 : -1);
  const int abs_score = std::abs(score);
  const int p =
      std::upper_bound(kScoreTable, kScoreTable + kScoreTableSize, abs_score) -
          kScoreTable;
  return sign * p;
}
}