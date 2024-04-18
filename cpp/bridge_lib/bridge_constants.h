//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_CONSTANTS_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_CONSTANTS_H_
#include "types.h"
#include <vector>
namespace rl::bridge {
using Cards = std::vector<Action>;
using DDT = std::vector<int>;

enum Denomination {
  kNoDenomination = -1, kClubs = 0, kDiamonds, kHearts, kSpades, kNoTrump
};
inline constexpr int kNumDenominations = 5;
constexpr char kDenominationChar[] = "CDHSN";

enum DoubleStatus {
  kUndoubled = 1, kDoubled = 2, kRedoubled = 4
};
inline constexpr int kNumDoubleStates = 3;

inline constexpr int kNumPlayers = 4;
constexpr char kPlayerChar[] = "NESW";

inline constexpr int kNumSuits = 4; // C,D,H,S
inline constexpr int kNumCardsPerSuit = 13; // 2,3,4,5,6,7,8,9,T,K,A
inline constexpr int kNumPartnerships = 2; // NS, EW
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

inline constexpr int kNumObservationTypes = 4; // Bid, lead, declare, defend
// Because bids always increase, any individual bid can be made at most once.
// Thus for each bid, we only need to track (a) who bid it (if anyone), (b) who
// doubled it (if anyone), and (c) who redoubled it (if anyone).
// We also report the number of passes before the first bid; we could
// equivalently report which player made the first call.
// This is much more compact than storing the auction call-by-call, which
// requires 318 turns * 38 possible calls per turn = 12084 bits (although
// in practice almost all auctions have fewer than 80 calls).
// eventually, kAuctionTensorSize=(4 * 106) + 52 + 4 = 480
inline constexpr int kAuctionTensorSize =
    kNumPlayers * (1          // Did this player pass before the opening bid?
        + kNumBids // Did this player make each bid?
        + kNumBids // Did this player double each bid?
        + kNumBids // Did this player redouble each bid?
    ) +
        kNumCards // Our hand
        + kNumVulnerabilities * kNumPartnerships;

// 52 * 3
inline constexpr int kHiddenInfoTensorSize = kNumCards * (kNumPlayers - 1);
inline constexpr int kPerfectInfoTensorSize = kHiddenInfoTensorSize + kAuctionTensorSize;
inline constexpr int kCardsTensorSize = kNumPlayers * kNumCards;
inline constexpr int kFinalTensorSize =
    kNumPlayers * kNumCards // Each player's hand
        + kNumBidLevels             // What the contract is
        + kNumDenominations         // What trumps are
        + kNumDoubleStates          // Undoubled / doubled / redoubled
        + kNumPlayers              // Who declarer is
        + kNumVulnerabilities;     // Vulnerability of the declaring side

// eventually, kPublicInfoTensorSize= 480 - 52+ 4 = 432
inline constexpr int kPublicInfoTensorSize =
    kAuctionTensorSize // The auction
        - kNumCards        // But not any player's cards
        + kNumPlayers;     // Plus trailing passes
// eventually, kPlayTensorSize = 7 + 5 + 3 + 4 + 2 + 52 + 52 + 4 * 52 + 4 * 52 +
// 13 + 13 = 567
inline constexpr int kPlayTensorSize =
    kNumBidLevels             // What the contract is
        + kNumDenominations       // What trumps are
        + kNumOtherCalls          // Undoubled / doubled / redoubled
        + kNumPlayers             // Who declarer is
        + kNumVulnerabilities     // Vulnerability of the declaring side
        + kNumCards               // Our remaining cards
        + kNumCards               // Dummy's remaining cards
        + kNumPlayers * kNumCards // Cards played to the previous trick
        + kNumPlayers * kNumCards // Cards played to the current trick
        + kNumTricks              // Number of tricks we have won
        + kNumTricks;             // Number of tricks they have won
// eventually, kObservationTensorSize = 480
inline constexpr int kObservationTensorSize = kAuctionTensorSize;
// every bid can lead a sequence as 1C-P-P-D-P-P-R-P-P
// eventually, kMaxAuctionLength = 35 * (1 + 8) + 4 = 319
inline constexpr int kMaxAuctionLength =
    kNumBids * (1 + kNumPlayers * 2) + kNumPlayers;
inline constexpr Player kFirstPlayer = 0;
inline constexpr int kDoubleDummyResultSize = kNumDenominations * kNumPlayers;
enum class Suit { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };

enum Calls { kPass = 0, kDouble = 1, kRedouble = 2 };

constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDHS";

constexpr char kLevelChar[] = "-1234567";

const std::vector<Suit> kAllSuits = {Suit::kClubs, Suit::kDiamonds, Suit::kHearts, Suit::kSpades};

}
#endif // BRIDGE_RESEARCH_CPP_BRIDGE_CONSTANTS_H_
