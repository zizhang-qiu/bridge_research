//
// Created by qzz on 2023/2/20.
//

#ifndef BRIDGE_RESEARCH_ENCODE_BRIDGE_H
#define BRIDGE_RESEARCH_ENCODE_BRIDGE_H

#include <vector>
#include "third_party/abseil-cpp/absl/types/span.h"
#include "third_party/abseil-cpp/absl/types/optional.h"

namespace bridge_encode {

enum Phase {
    kDeal = 0,
    kAuction = 1,
    kGameOver = 2
};

enum Calls {
    kPass = 0,
    kDouble = 1,
    kRedouble = 2
};

enum class Suit {
    kClubs = 0,
    kDiamonds = 1,
    kHearts = 2,
    kSpades = 3
};

enum Denomination {
    kClubs = 0,
    kDiamonds = 1,
    kHearts = 2,
    kSpades = 3,
    kNoTrump = 4
};


inline constexpr int kNumDoubleStates = 3;
inline constexpr int kNumPlayers = 4;
inline constexpr int kNumDenominations = 5;
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
inline constexpr int kFirstBid = kRedouble + 1;
inline constexpr int kBiddingActionBase = kNumCards;
inline constexpr int kAuctionTensorSize =
        kNumPlayers * (1           // Did this player pass before the opening bid?
                       + kNumBids  // Did this player make each bid?
                       + kNumBids  // Did this player double each bid?
                       + kNumBids  // Did this player redouble each bid?
        ) +
        kNumCards                                  // Our hand
        + kNumVulnerabilities * kNumPartnerships;


// functions for partner
int GetPartnership(int player) {
    return player & 1;
}

int GetPartner(int player) {
    return player ^ 2;
}

// functions for bid,
int GetBid(int level, Denomination denomination) {
    return (level - 1) * kNumDenominations + denomination + kFirstBid;
}

int GetBidLevel(int bid) {
    return 1 + (bid - kNumOtherCalls) / kNumDenominations;
}

Denomination GetBidSuit(int bid) {
    return Denomination((bid - kNumOtherCalls) % kNumDenominations);
}

// Cards are represented as rank * kNumSuits + suit.
Suit GetCardSuit(int card) { return Suit(card % kNumSuits); }

int GetCardRank(int card) { return card / kNumSuits; }

//0=2c, 1=2d, 2=2h, 3=2s
int GetCard(Suit suit, int rank) {
    return rank * kNumSuits + static_cast<int>(suit);
}


//the auction observation tensor is made up by
//vulnerabilities (2 * 2)
//and for each player:
//Did this player pass before the opening bid? (1)
//Did this player make each bid? (35)
//Did this player double each bid? (35)
//Did this player redouble each bid? (35)
//current player's hand (52)
//4 + 52 + 4 * (3 * 35 + 1) = 480
std::vector<float> Encode(int player, bool is_dealer_vulnerable,
                          bool is_non_dealer_vulnerable, std::vector<int> holder,
                          std::vector<int> history){

    bool is_vulnerable[kNumPartnerships] = {is_dealer_vulnerable, is_non_dealer_vulnerable};
    std::vector<float> observation_tensor(kAuctionTensorSize);
    absl::Span<float> values = absl::MakeSpan(observation_tensor);
    std::fill(values.begin(), values.end(), 0.0);
    int partnership = GetPartnership(player);
    auto ptr = values.begin();
    ptr[is_vulnerable[partnership]] = 1;
    ptr += kNumVulnerabilities;
    ptr[is_vulnerable[1 - partnership]] = 1;
    ptr += kNumVulnerabilities;
    int last_bid = 0;
    for (int i = kNumCards; i < history.size(); ++i) {
        int this_call = history[i] - kBiddingActionBase;
        int relative_bidder = (i + kNumPlayers - player) % kNumPlayers;
        if (last_bid == 0 && this_call == kPass) ptr[relative_bidder] = 1;
        if (this_call == kDouble) {
            ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
                kNumPlayers + relative_bidder] = 1;
        } else if (this_call == kRedouble) {
            ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
                kNumPlayers * 2 + relative_bidder] = 1;
        } else if (this_call != kPass) {
            last_bid = this_call;
            ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
                relative_bidder] = 1;
        }
    }
    ptr += kNumPlayers * (1 + 3 * kNumBids);
    for (int i = 0; i < kNumCards; ++i)
        if (holder[i] == player) ptr[i] = 1;
//    ptr += kNumCards;

    return observation_tensor;
}



}
#endif //BRIDGE_RESEARCH_ENCODE_BRIDGE_H
