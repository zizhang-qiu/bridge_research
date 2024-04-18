//
// Created by qzz on 2023/7/6.
//
#include "encoder.h"
namespace rl::bridge {

std::vector<Player> OtherPlayersVector(const Player player) {
  std::vector<Player> ret;
  ret.reserve(kNumPlayers - 1);
  for (int interval = 1; interval < kNumPlayers; ++interval) {
    ret.push_back((player + interval) % kNumPlayers);
  }
  return ret;
}

std::vector<float> CanonicalObservationEncoder::EncodeAllHands(const std::shared_ptr<BridgeBiddingState> &state) {
  std::vector<float> hands_encoding(kAllHandsTensorSize);
  auto values = utils::Span<float>(hands_encoding);
  auto ptr = values.begin();
  const auto holder = state->GetHolder();
  for (const Player pl : {kNorth, kEast, kSouth, kWest}) {
    for (int card = 0; card < kNumCards; ++card) {
      if (holder[card].value() == pl) {
        ptr[card] = 1;
      }
    }
    ptr += kNumCards;
  }
  RL_CHECK_EQ(std::distance(values.begin(), ptr), kAllHandsTensorSize);
  RL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  return hands_encoding;
}

std::vector<float> CanonicalObservationEncoder::EncodeHistory(const std::shared_ptr<BridgeBiddingState> &state) {
  std::vector<float> history_encoding(kHistoryTensorSize);
  auto values = utils::Span<float>(history_encoding);
  auto ptr = values.begin();

  const auto full_history = state->FullHistory();
  int last_bid = 0;
  for (int i = kNumCards; i < full_history.size(); ++i) {
    int this_call = full_history[i].action;
    int this_bidder = i % kNumPlayers;
    // opening pass (0-3)
    if (last_bid == 0 && this_call == kPass) {
      ptr[this_bidder] = 1;
    }

    if (this_call == kDouble) {
      // double history (144-283)
      ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
          kNumPlayers + this_bidder] = 1;
    } else if (this_call == kRedouble) {
      // redouble history (284-423)
      ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
          kNumPlayers * 2 + this_bidder] = 1;
    } else if (this_call != kPass) {
      // bid history (4-143)
      last_bid = this_call;
      ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
          this_bidder] = 1;
    }
  }
  ptr += kNumPlayers * (1 + 3 * kNumBids);
  RL_CHECK_EQ(std::distance(values.begin(), ptr), kHistoryTensorSize);
  RL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  return history_encoding;
}

std::vector<float> CanonicalObservationEncoder::EncodeOwnHand(const std::shared_ptr<BridgeBiddingState> &state) {
  std::vector<float> own_hand_encoding(kNumCards);
  auto values = utils::Span<float>(own_hand_encoding);
  auto ptr = values.begin();
  const Player current_player = state->CurrentPlayer();
  const auto holder = state->GetHolder();
  for (int card = 0; card < kNumCards; ++card) {
    if (holder[card].value() == current_player) {
      ptr[card] = 1;
    }
  }
  ptr += kNumCards;
  RL_CHECK_EQ(std::distance(values.begin(), ptr), kNumCards);
  RL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  return own_hand_encoding;
}
std::vector<float> CanonicalObservationEncoder::EncodeOtherHands(const std::shared_ptr<BridgeBiddingState> &state) {
  std::vector<float> other_hand_encoding(kOtherHandsTensorSize);
  auto values = utils::Span<float>(other_hand_encoding);
  auto ptr = values.begin();
  const Player current_player = state->CurrentPlayer();
  const std::vector<int> other_players_vec = OtherPlayersVector(current_player);
  for(const auto pl:other_players_vec){

  }
  return {};
}
}