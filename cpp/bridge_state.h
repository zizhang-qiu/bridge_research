//
// Created by qzz on 2023/2/25.
//

#ifndef BRIDGE_RESEARCH_BRIDGE_STATE_H
#define BRIDGE_RESEARCH_BRIDGE_STATE_H
#include "bridge_deal.h"
#include "bridge_scoring.h"
#include "rl/logging.h"
#include "span.h"
#include "str_utils.h"
#include "third_party/dds/include/dll.h"
#include "rl/types.h"
#include "rl/utils.h"
#include <array>
#include <sstream>
#include <string>
#include <utility>
namespace rl::bridge {

inline constexpr int kFirstBid = kRedouble + 1;

int Bid(int level, Denomination denomination);

int BidLevel(int bid);

Denomination BidTrump(int bid);

// Cards are represented as rank * kNumSuits + suit.
Suit CardSuit(int card);

int CardRank(int card);

// 0=2c, 1=2d, 2=2h, 3=2s
int Card(Suit suit, int rank);

std::string CardString(int card);

std::string BidString(int bid);

// There are two partnerships: players 0 and 2 versus players 1 and 3.
// We call 0 and 2 partnership 0, and 1 and 3 partnership 1.
int Partnership(Player player);

int Partner(Player player);

int SuitToDDSStrain(Suit suit);

int DenominationToDDSStrain(Denomination denomination);

class BridgeBiddingState {
 public:
  explicit BridgeBiddingState(const BridgeDeal &deal)
      : is_vulnerable_{deal.is_dealer_vulnerable, deal.is_non_dealer_vulnerable} {
    RL_CHECK_EQ(deal.cards.size(), kNumCards);
    RL_CHECK_LT(deal.dealer, kNumCards);
    RL_CHECK_GE(deal.dealer, 0);
    auto cards = deal.cards;
    dealer_ = deal.dealer;
    current_player_ = deal.dealer;
    InitVulStrs();
    GetHolder(cards);
    if (deal.ddt.has_value()) {
      ConvertDoubleDummyResults(deal.ddt.value());
    }
//    ComputeDoubleDummyResult();
  }

//  BridgeBiddingState(Player dealer, const Cards &cards,
//                     bool is_dealer_vulnerable, bool is_non_dealer_vulnerable,
//                     const DDT &double_dummy_table)
//      : is_vulnerable_{is_dealer_vulnerable, is_non_dealer_vulnerable} {
//    RL_CHECK_EQ(cards.size(), kNumCards);
//    RL_CHECK_EQ(double_dummy_table.size(), kDoubleDummyResultSize);
//    RL_CHECK_LT(dealer, kNumCards);
//    RL_CHECK_GE(dealer, 0);
//    dealer_ = dealer;
//    current_player_ = dealer;
//    InitVulStrs();
//    GetHolder(cards);
//    ConvertDoubleDummyResults(double_dummy_table);
//  }

  Player CurrentPlayer() const { return current_player_; }
  int CurrentPhase() const { return phase_; }
  bool Terminated() const { return phase_ == kGameOver; }
  void Terminate() { phase_ = Phase::kGameOver; }
  void InitVulStrs();
  std::vector<Action> History() const;
  std::vector<PlayerAction> FullHistory() const { return history_; }
  std::vector<Action> BidHistory() const;
  void ApplyAction(Action action);
  std::string BidStr() const;
  std::vector<std::string> BidStrHistory() const;
  std::vector<double> Returns() const;
  std::string ContractString() const { return contract_.ToString(); }
  std::string ObservationString(Player player) const;
  std::string ToString() const;
  std::vector<float> ObservationTensor(Player player) const;
  std::vector<float> ObservationTensor() const;
  std::vector<float> ObservationTensor2() const;
  std::vector<Action> LegalActions() const;
  std::vector<float> LegalActionsMask() const;
  std::shared_ptr<BridgeBiddingState> Clone() const;

  std::vector<int> GetDoubleDummyTable();
  // use dds to compute double dummy result
  void ComputeDoubleDummyResult();

  std::vector<int> GetActualTrickAndDDTrick() {
    if (!double_dummy_results_.has_value()) {
      ComputeDoubleDummyResult();
    }
    auto trump = contract_.trumps;
    auto level = contract_.level;
    int actual_trick = level + 6;
    std::vector<int> ret = {actual_trick, num_declarer_tricks_};
    return ret;
  }

  std::vector<Action> GetPlayerCards(Player player) const;
  Contract GetContract() const { return contract_; }
  static std::vector<int> ObservationTensorShape() { return {kAuctionTensorSize}; }

  static int ObservationTensorSize() {
    std::vector<int> shape = ObservationTensorShape();
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<int>());
  }

  friend std::ostream &
  operator<<(std::ostream &os,
             const std::shared_ptr<BridgeBiddingState> &state) {
    return os << state->ToString();
  }

 private:
  enum Phase { kAuction = 0, kGameOver = 1 };
  int num_passes_ = 0; // Number of consecutive passes since the last non-pass.
  int num_declarer_tricks_ = 0;
  std::vector<double> returns_ = std::vector<double>(kNumPlayers);
  std::vector<PlayerAction> history_;
  bool is_vulnerable_[kNumPartnerships]{};
  Phase phase_ = Phase::kAuction;
  Contract contract_{0};
  // dealer is the first player to bid
  Player dealer_;
  mutable std::optional<ddTableResults> double_dummy_results_{};
  Player current_player_;
  // first_bidder tracks for each suit, which player in his partnership bid
  // first.
  std::array<std::array<std::optional<Player>, kNumDenominations>,
             kNumPartnerships> first_bidder_{};
  // holder tracks each card's owner.
  std::array<std::optional<Player>, kNumCards> holder_{};
  std::string vul_strs[2][2] = {{"None", ""}, {"", "All"}};

  void GetHolder(const Cards &cards);
  void ScoreUp();
  // convert ddt
  void ConvertDoubleDummyResults(const std::vector<int> &double_dummy_table);
  void ApplyBiddingAction(Action call);
  std::array<std::string, kNumSuits>
  FormatHand(int player, bool mark_voids,
             const std::array<std::optional<Player>, kNumCards> &deal) const;
  std::string FormatDeal() const;
  std::string FormatVulnerability() const;
  std::string FormatAuction(bool trailing_query) const;
  std::string FormatResult() const;
  void WriteObservationTensor(Player player, utils::Span<float> values) const;
  std::vector<Action> BiddingLegalActions() const;
};

} // namespace rl::bridge
#endif // BRIDGE_RESEARCH_BRIDGE_STATE_H
