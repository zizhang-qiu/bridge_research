//
// Created by qzz on 2023/2/25.
//

#ifndef BRIDGE_RESEARCH_BRIDGE_STATE_H
#define BRIDGE_RESEARCH_BRIDGE_STATE_H
#include "bridge_deal.h"
#include "bridge_scoring.h"
#include "logging.h"
#include "span.h"
#include "str_utils.h"
#include "third_party/dds/include/dll.h"
#include "types.h"
#include "utils.h"
#include <array>
#include <sstream>
#include <string>
#include <utility>
namespace rl::bridge {


inline constexpr int kFirstBid = kRedouble + 1;

int Bid(int level, Denomination denomination) {
  return (level - 1) * kNumDenominations + denomination + kFirstBid;
}

int BidLevel(int bid) { return 1 + (bid - kNumOtherCalls) / kNumDenominations; }

Denomination BidTrump(int bid) {
  return Denomination((bid - kNumOtherCalls) % kNumDenominations);
}

// Cards are represented as rank * kNumSuits + suit.
Suit CardSuit(int card) { return Suit(card % kNumSuits); }

int CardRank(int card) { return card / kNumSuits; }

// 0=2c, 1=2d, 2=2h, 3=2s
int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}



std::string CardString(int card) {
  return {kSuitChar[static_cast<int>(CardSuit(card))],
          kRankChar[CardRank(card)]};
}

std::string BidString(int bid) {
  if (bid == kPass)
    return "Pass";
  if (bid == kDouble)
    return "Dbl";
  if (bid == kRedouble)
    return "RDbl";
  return {kLevelChar[BidLevel(bid)], kDenominationChar[BidTrump(bid)]};
}

// There are two partnerships: players 0 and 2 versus players 1 and 3.
// We call 0 and 2 partnership 0, and 1 and 3 partnership 1.
int Partnership(Player player) { return player & 1; }

int Partner(Player player) { return player ^ 2; }

int SuitToDDSStrain(Suit suit) {
  return 3 - static_cast<int>(suit);
}

int DenominationToDDSStrain(Denomination denomination) {
  return denomination == kNoTrump ? denomination : 3 - denomination;
}



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

  void InitVulStrs() {
    auto dealer_partnership = Partnership(dealer_);
    if (dealer_partnership == 0) {
      vul_strs[1][0] = "N/S";
      vul_strs[0][1] = "E/W";
    } else {
      vul_strs[1][0] = "E/W";
      vul_strs[0][1] = "N/S";
    }
  }

  std::vector<Action> History() const {
    std::vector<Action> history;
    history.reserve(history_.size());
    for (const auto player_action : history_) {
      history.emplace_back(player_action.action);
    }
    return history;
  }

  std::vector<PlayerAction> FullHistory() const { return history_; }

  std::vector<Action> BidHistory() const {
    RL_CHECK_GE(phase_, Phase::kAuction);
    std::vector<Action> bid_history;
    for (int i = kNumCards; i < history_.size(); i++) {
      bid_history.emplace_back(history_[i].action);
    }
    return bid_history;
  }

  void Terminate() { phase_ = Phase::kGameOver; }

  void ApplyAction(Action action) {
    RL_CHECK_EQ(phase_, Phase::kAuction);
    //        RL_CHECK_TRUE(utils::IsValueInVector(LegalActions(), action));
    history_.emplace_back(PlayerAction{current_player_, action});
    ApplyBiddingAction(action);
  }

  std::string BidStr() const {
    RL_CHECK_GE(phase_, Phase::kAuction);
    std::string bid_str =
        utils::StrFormat("Dealer is %c, ", kPlayerChar[dealer_]);
    for (int i = kNumCards; i < history_.size(); i++) {
      bid_str += BidString(history_[i].action);
      bid_str += ", ";
    }
    return bid_str;
  }

  std::vector<std::string> BidStrHistory() {
    RL_CHECK_GE(phase_, Phase::kAuction);
    std::vector<std::string> bid_str_history;
    for (int i = kNumCards; i < history_.size(); i++) {
      bid_str_history.emplace_back(BidString(history_[i].action));
    }
    return bid_str_history;
  }

  bool Terminated() const { return phase_ == kGameOver; }

  std::vector<double> Returns() const {
    RL_CHECK_EQ(phase_, kGameOver);
    return returns_;
  }

  std::string ContractString() const { return contract_.ToString(); }

  std::string ObservationString(Player player) const {
    RL_CHECK_GE(player, 0);
    RL_CHECK_LT(player, kNumPlayers);
    if (Terminated())
      return ToString();
    std::string rv = FormatVulnerability();
    auto cards = FormatHand(player, /*mark_voids=*/true, holder_);
    for (int suit = kNumSuits - 1; suit >= 0; --suit)
      utils::StrAppend(&rv, cards[suit], "\n");
    if (history_.size() > kNumCards)
      utils::StrAppend(
          &rv, FormatAuction(/*trailing_query=*/phase_ == Phase::kAuction &&
              player == current_player_));
    return rv;
  }

  std::string ToString() const {
    std::string rv = utils::StrCat(FormatVulnerability(), FormatDeal());
    if (history_.size() > kNumCards)
      utils::StrAppend(&rv, FormatAuction(/*trailing_query=*/false));
    if (Terminated())
      utils::StrAppend(&rv, FormatResult());
    return rv;
  }

  std::vector<float> ObservationTensor(Player player) {
    RL_CHECK_GE(player, 0);
    RL_CHECK_LT(player, kNumPlayers);
    std::vector<float> observation(ObservationTensorSize());
    WriteObservationTensor(player, utils::Span<float>(observation));
    return observation;
  }

  std::vector<float> ObservationTensor() {
    return ObservationTensor(current_player_);
  }

  std::vector<Action> LegalActions() {
    if (Terminated()) {
      std::vector<Action> legal_actions;
      return legal_actions;
    }
    return BiddingLegalActions();
  }

  std::shared_ptr<BridgeBiddingState> Clone() const {
    return std::make_shared<BridgeBiddingState>(*this);
  }

  std::vector<int> GetDoubleDummyTable() {
    if (!double_dummy_results_.has_value()) {
      ComputeDoubleDummyResult();
    }
    auto double_dummy_results = double_dummy_results_->resTable;
    std::vector<int> ret(kDoubleDummyResultSize);
    for (auto denomination : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      for (auto player : {kNorth, kEast, kSouth, kWest}) {
        ret[denomination * kNumPlayers + player] =
            double_dummy_results[denomination][player];
      }
    }
    return ret;
  }

  static std::vector<int> ObservationTensorShape() {
    return {kAuctionTensorSize};
  }

  static int ObservationTensorSize() {
    std::vector<int> shape = ObservationTensorShape();
    return shape.empty() ? 0
                         : std::accumulate(shape.begin(), shape.end(), 1,
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
  // dealer is the first player to and bid
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

  void GetHolder(const Cards &cards) {
    for (int i = 0; i < kNumCards; i++) {
      auto card = cards[i];
      auto player_action = PlayerAction{kChancePlayerId, card};
      history_.emplace_back(player_action);
      Player card_holder = i % kNumPlayers;
      //            std::cout<< card_holder << std::endl;
      holder_[card] = card_holder;
    }
  }

  void ScoreUp() {
    //        std::cout << "declarer: " << contract_.declarer << std::endl;
    int declarer_score = Score(contract_, num_declarer_tricks_,
                               is_vulnerable_[Partnership(contract_.declarer)]);
    for (int pl = 0; pl < kNumPlayers; ++pl) {
      returns_[pl] = Partnership(pl) == Partnership(contract_.declarer)
                     ? declarer_score
                     : -declarer_score;
    }
  }

  // convert ddt
  void ConvertDoubleDummyResults(const std::vector<int> &double_dummy_table) {

    auto double_dummy_results = ddTableResults{};
    for (auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      for (auto player : {kNorth, kEast, kSouth, kWest}) {
        auto index = trump * kNumPlayers + player;
        //                std::cout << "index: " << index << std::endl;
        //                std::cout << "value: " << ptr[index] << std::endl;
        double_dummy_results.resTable[trump][player] =
            double_dummy_table[index];
      }
    }
    double_dummy_results_ = double_dummy_results;
    //        std::cout << "double_dummy_results_.has_value()=" <<
    //        double_dummy_results_.has_value() << std::endl;
  }

  // use dds to compute double dummy result
  void ComputeDoubleDummyResult() {
    //        RL_CHECK_TRUE(!double_dummy_results_.has_value());
    //    absl::MutexLock lock(&dds_mutex);
    double_dummy_results_ = ddTableResults{};
    ddTableDeal dd_table_deal{};
    for (int suit = 0; suit < kNumSuits; ++suit) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        const int player = holder_[Card(Suit(suit), rank)].value();
        dd_table_deal.cards[player][suit] += 1 << (2 + rank);
      }
    }
    SetMaxThreads(0);
    const int return_code =
        CalcDDtable(dd_table_deal, &double_dummy_results_.value());
    if (return_code != RETURN_NO_FAULT) {
      char error_message[80];
      ErrorMessage(return_code, error_message);
      std::cerr << utils::StrCat("double_dummy_solver:", error_message)
                << std::endl;
    }
  }

  void ApplyBiddingAction(int call) {
    if (call == kPass) {
      ++num_passes_;
    } else {
      num_passes_ = 0;
    }
    auto partnership = Partnership(current_player_);
    if (call == kDouble) {
      // partner can't double the bid
      RL_CHECK_NE(Partnership(contract_.declarer), partnership);
      // can't double a bid if it is doubled
      RL_CHECK_EQ(contract_.double_status, kUndoubled);
      // can't double if there is no bid
      RL_CHECK_GT(contract_.level, 0);
      contract_.double_status = kDoubled;
    } else if (call == kRedouble) {
      // player can only redouble partner's bid
      RL_CHECK_EQ(Partnership(contract_.declarer), partnership);
      // player can only redouble if thr bid is doubled
      RL_CHECK_EQ(contract_.double_status, kDoubled);
      contract_.double_status = kRedoubled;
    } else if (call == kPass) {
      // check if the game is passed out
      if (num_passes_ == 4) {
        phase_ = kGameOver;
      } else if (num_passes_ == 3 && contract_.level > 0) {
        phase_ = kGameOver;
        if(!double_dummy_results_.has_value()){
          ComputeDoubleDummyResult();
        }
        num_declarer_tricks_ =
            double_dummy_results_
                ->resTable[contract_.trumps][contract_.declarer];
        ScoreUp();
      }
    } else {
      // A bid was made
      // the bid should be higher in trump or level
      RL_CHECK_TRUE((BidLevel(call) > contract_.level) ||
          (BidLevel(call) == contract_.level &&
              BidTrump(call) > contract_.trumps))
      contract_.level = BidLevel(call);
      contract_.trumps = BidTrump(call);
      contract_.double_status = kUndoubled;
      if (!first_bidder_[partnership][contract_.trumps].has_value()) {
        // Partner cannot declare this denomination.
        first_bidder_[partnership][contract_.trumps] = current_player_;
      }
      contract_.declarer = first_bidder_[partnership][contract_.trumps].value();
    }

    current_player_ = (current_player_ + 1) % kNumPlayers;
  }

  std::array<std::string, kNumSuits>
  FormatHand(int player, bool mark_voids,
             const std::array<std::optional<Player>, kNumCards> &deal) const {
    std::array<std::string, kNumSuits> cards_str_per_suit;
    for (int suit = 0; suit < kNumSuits; ++suit) {
      cards_str_per_suit[suit].push_back(kSuitChar[suit]);
      cards_str_per_suit[suit].push_back(' ');
      bool is_void = true;
      for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
        if (player == deal[Card(Suit(suit), rank)]) {
          cards_str_per_suit[suit].push_back(kRankChar[rank]);
          is_void = false;
        }
      }
      if (is_void && mark_voids)
        utils::StrAppend(&cards_str_per_suit[suit], "none");
    }
    return cards_str_per_suit;
  }

  std::string FormatDeal() const {
    std::array<std::array<std::string, kNumSuits>, kNumPlayers> cards_str_per_suit_player;

    for (auto player : {kNorth, kEast, kSouth, kWest}) {
      cards_str_per_suit_player[player] = FormatHand(player, /*mark_voids=*/false, holder_);
    }

    constexpr int kColumnWidth = 8;
    std::string padding(kColumnWidth, ' ');
    std::string rv;
    for (int suit = kNumSuits - 1; suit >= 0; --suit)
      utils::StrAppend(&rv, padding, cards_str_per_suit_player[kNorth][suit], "\n");
    for (int suit = kNumSuits - 1; suit >= 0; --suit)
      utils::StrAppend(&rv,
                       utils::StrFormat("%-8s", cards_str_per_suit_player[kWest][suit].c_str()),
                       padding, cards_str_per_suit_player[kEast][suit], "\n");
    for (int suit = kNumSuits - 1; suit >= 0; --suit)
      utils::StrAppend(&rv, padding, cards_str_per_suit_player[kSouth][suit], "\n");
    return rv;
  }

  std::string FormatVulnerability() const {
    std::string vul_str = "Vul: ";
    utils::StrAppend(&vul_str, vul_strs[is_vulnerable_[0]][is_vulnerable_[1]],
                     "\n");
    return vul_str;
  }

  std::string FormatAuction(bool trailing_query) const {
    RL_CHECK_GT(history_.size(), kNumCards);
    std::string rv = "\nWest  North East  South\n";
    int num_paddings = dealer_ == kWest ? 0 : (dealer_ % kWest + 1) * 6;
    //        std::cout << "num_paddings: " << num_paddings << std::endl;
    std::string padding(num_paddings, ' ');
    utils::StrAppend(&rv, padding);
    for (int i = kNumCards; i < history_.size(); ++i) {
      if (i % kNumPlayers == kNumPlayers - 1 - dealer_ && i > kNumCards) {
        rv.push_back('\n');
      }
      utils::StrAppend(
          &rv, utils::StrFormat("%-6s", BidString(history_[i].action).c_str()));
    }
    if (trailing_query) {
      if ((history_.size()) % kNumPlayers == kNumPlayers - 1)
        rv.push_back('\n');
      rv.push_back('?');
    }
    return rv;
  }

  std::string FormatResult() const {
    RL_CHECK_TRUE(Terminated())
    std::string rv;
    if (contract_.level) {
      utils::StrAppend(&rv, "\n\nDeclarer tricks: ", num_declarer_tricks_);
    }
    utils::StrAppend(&rv, "\nScore: N/S ", returns_[kNorth], " E/W ",
                     returns_[kEast]);
    return rv;
  }

  void WriteObservationTensor(Player player, utils::Span<float> values) const {
    std::fill(values.begin(), values.end(), 0.0);
    int partnership = Partnership(player);
    auto ptr = values.begin();

    /*
    vulnerability encoding takes 4 bits,
    the first 2 bits represents whether the first partnership(i.e. NS) is
    vulnerable, [0, 1] represent vulnerable and [1,0] represent non-vulnerable,
    same for the next 2 bits represent second partnership(i.e. EW)
    */
    ptr[is_vulnerable_[partnership]] = 1;
    ptr += kNumVulnerabilities;
    ptr[is_vulnerable_[1 - partnership]] = 1;
    ptr += kNumVulnerabilities;

    int last_bid = 0;
    for (int i = kNumCards; i < history_.size(); ++i) {
      int this_call = history_[i].action;
      /*
      relative bidder is defined by clockwise, in current player's perspectives
      suppose dealer is North.
      if i=52 and player is 0(North), then relative bidder is 0
      if i=52 and player is 1(East), then relative bidder is 3(because start
      from East, North is the third player) if i=52 and player is 2(South), then
      relative bidder is 2 if i=52 and player is 3(West), then relative bidder
      is 1 if i=53 and player is 0(North), then relative bidder is 1
      */
      int this_call_bidder = (i + dealer_) % kNumPlayers;
      int relative_bidder =
          (this_call_bidder >= player ? this_call_bidder
                                      : this_call_bidder + kNumPlayers) -
              player;
      //            std::cout << "relative_bidder:" << relative_bidder <<
      //            std::endl;
      // opening pass
      if (last_bid == 0 && this_call == kPass) {
        ptr[relative_bidder] = 1;
      }

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

    // player's cards
    ptr += kNumPlayers * (1 + 3 * kNumBids);
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == player) {
        ptr[i] = 1;
      }
    ptr += kNumCards;
    RL_CHECK_EQ(std::distance(values.begin(), ptr), kAuctionTensorSize);
    RL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  }

  std::vector<Action> BiddingLegalActions() const {
    std::vector<Action> legal_actions;
    legal_actions.reserve(kNumCalls);
    legal_actions.emplace_back(kPass);
    if (contract_.level > 0 &&
        Partnership(contract_.declarer) != Partnership(current_player_) &&
        contract_.double_status == kUndoubled) {
      legal_actions.emplace_back(kDouble);
    }
    if (contract_.level > 0 &&
        Partnership(contract_.declarer) == Partnership(current_player_) &&
        contract_.double_status == kDoubled) {
      legal_actions.emplace_back(kRedouble);
    }
    for (int bid = Bid(contract_.level, contract_.trumps) + 1; bid < kNumCalls;
         ++bid) {
      legal_actions.emplace_back(bid);
    }
    return legal_actions;
  }
};

} // namespace rl::bridge
#endif // BRIDGE_RESEARCH_BRIDGE_STATE_H
