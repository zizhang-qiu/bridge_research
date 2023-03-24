//
// Created by qzz on 2023/2/23.
//
#include <array>
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "string"
#include "utils.h"
#include "logging.h"
#ifndef BRIDGE_RESEARCH_BRIDGE_SCORING_H
#define BRIDGE_RESEARCH_BRIDGE_SCORING_H
namespace rl::bridge{
enum Denomination {
    kClubs = 0, kDiamonds, kHearts, kSpades, kNoTrump
};
inline constexpr int kNumDenominations = 5;
constexpr char kDenominationChar[] = "CDHSN";

enum DoubleStatus {
    kUndoubled = 1, kDoubled = 2, kRedoubled = 4
};
inline constexpr int kNumDoubleStates = 3;

inline constexpr int kNumPlayers = 4;
constexpr char kPlayerChar[] = "NESW";

inline constexpr int kNumSuits = 4; //C,D,H,S
inline constexpr int kNumCardsPerSuit = 13; //2,3,4,5,6,7,8,9,T,K,A
inline constexpr int kNumPartnerships = 2; //NS, EW
inline constexpr int kNumBidLevels = 7;   // Bids can be from 7 to 13 tricks.
inline constexpr int kNumOtherCalls = 3;  // Pass, Double, Redouble
inline constexpr int kNumVulnerabilities = 2;  // Vulnerable or non-vulnerable.
inline constexpr int kNumBids = kNumBidLevels * kNumDenominations; // 1C,1D,...7H,7S,7NT
inline constexpr int kNumCalls = kNumBids + kNumOtherCalls; // 35+ 3
inline constexpr int kNumCards = kNumSuits * kNumCardsPerSuit; // 52 cards
inline constexpr int kNumCardsPerHand = kNumCards / kNumPlayers; // 13 cards for every player
inline constexpr int kNumTricks = kNumCardsPerHand; // 13 tricks can be done
inline constexpr int kMaxScore = 7600;  // See http://www.rpbridge.net/2y66.htm
inline constexpr int kMaxImp = 24;

struct Contract {
    int level = 0;
    Denomination trumps = kNoTrump;
    DoubleStatus double_status = kUndoubled;
    int declarer = -1;

    std::string ToString() const{
        if (level == 0) return "Passed Out";
        std::string str = absl::StrCat(level, std::string{kDenominationChar[trumps]});
        if (double_status == kDoubled) absl::StrAppend(&str, "X");
        if (double_status == kRedoubled) absl::StrAppend(&str, "XX");
        absl::StrAppend(&str, " ", std::string{kPlayerChar[declarer]});
        return str;
    }

    int Index() const{
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
};
constexpr int kBaseTrickScores[] = {20, 20, 30, 30, 30};

int ScoreContract(Contract contract, DoubleStatus double_status) {
    int score = contract.level * kBaseTrickScores[contract.trumps];
    if (contract.trumps == kNoTrump) score += 10;
    return score * double_status;
}

// Score for failing to make the contract (will be negative).
int ScoreUndertricks(int undertricks, bool is_vulnerable,
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

// Score for tricks made in excess of the bid.
int ScoreOvertricks(Denomination trump_suit, int overtricks, bool is_vulnerable,
                    DoubleStatus double_status) {
    if (double_status == kUndoubled) {
        return overtricks * kBaseTrickScores[trump_suit];
    } else {
        return (is_vulnerable ? 100 : 50) * overtricks * double_status;
    }
}

// Bonus for making a doubled or redoubled contract.
int ScoreDoubledBonus(DoubleStatus double_status) {
    return 50 * (double_status / 2);
}

// Bonuses for partscore, game, or slam.
int ScoreBonuses(int level, int contract_score, bool is_vulnerable) {
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

int Score(Contract contract, int declarer_tricks, bool is_vulnerable){
    if (contract.level == 0) return 0;
    int contracted_tricks = 6 + contract.level;
    int contract_result = declarer_tricks - contracted_tricks;
    if (contract_result < 0) {
        return ScoreUndertricks(-contract_result, is_vulnerable,
                                contract.double_status);
    } else {
        int contract_score = ScoreContract(contract, contract.double_status);
        int bonuses = ScoreBonuses(contract.level, contract_score, is_vulnerable) +
                      ScoreDoubledBonus(contract.double_status) +
                      ScoreOvertricks(contract.trumps, contract_result,
                                      is_vulnerable, contract.double_status);
        return contract_score + bonuses;
    }
}

// All possible contracts.
inline constexpr int kNumContracts =
        kNumBids * kNumPlayers * kNumDoubleStates + 1;

constexpr std::array<Contract, kNumContracts> AllContracts() {
    std::array<Contract, kNumContracts> contracts;
    int i = 0;
    contracts[i++] = Contract();
    for (int level: {1, 2, 3, 4, 5, 6, 7}) {
        for (Denomination trumps:
                {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
            for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
                for (DoubleStatus double_status: {kUndoubled, kDoubled, kRedoubled}) {
                    contracts[i++] = Contract{level, trumps, double_status, declarer};
                }
            }
        }
    }
    return contracts;
}

inline constexpr std::array<Contract, kNumContracts> kAllContracts =
        AllContracts();

inline int GetImp(int my, int other){
    RL_CHECK_GE(my, -kMaxScore);
    RL_CHECK_GE(other, -kMaxScore);
    RL_CHECK_LE(my, kMaxScore);
    RL_CHECK_LE(other, kMaxScore);
    std::vector<int> imp_table = {15, 45, 85, 125, 165, 215, 265, 315, 365, 425, 495, 595, 745, 895,
                                  1095, 1295, 1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995};
    int abs_diff = std::abs(my - other);
    int index = std::distance(imp_table.begin(),
                              std::lower_bound(imp_table.begin(), imp_table.end(), abs_diff));
    return index * ((my > other) ? 1 : -1);
}
}
#endif //BRIDGE_RESEARCH_BRIDGE_SCORING_H
