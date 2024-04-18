//
// Created by qzz on 2023/7/6.
//

#ifndef BRIDGE_RESEARCH_PYSRC_ENCODER_H_
#define BRIDGE_RESEARCH_PYSRC_ENCODER_H_
#include "bridge_state.h"
namespace rl::bridge {

inline constexpr int kAllHandsTensorSize = kNumCards * kNumPlayers;
inline constexpr int kOtherHandsTensorSize = kNumCards * (kNumPlayers - 1);
inline constexpr int kHistoryTensorSize =
    kNumPlayers * (1          // Did this player pass before the opening bid?
        + kNumBids // Did this player make each bid?
        + kNumBids // Did this player double each bid?
        + kNumBids // Did this player redouble each bid?
    );
class CanonicalObservationEncoder {
 public:
  CanonicalObservationEncoder() = default;

  std::vector<float> EncodeAllHands(const std::shared_ptr<BridgeBiddingState> &state);

  std::vector<float> EncodeHistory(const std::shared_ptr<BridgeBiddingState> &state);

  std::vector<float> EncodeOwnHand(const std::shared_ptr<BridgeBiddingState> &state);

  std::vector<float> EncodeOtherHands(const std::shared_ptr<BridgeBiddingState> &state);
};
}
#endif //BRIDGE_RESEARCH_PYSRC_ENCODER_H_
