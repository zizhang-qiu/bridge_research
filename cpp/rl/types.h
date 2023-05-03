//
// Created by qzz on 2022/9/20.
//

#ifndef BRIDGE_RESEARCH_RL_TYPES_H
#define BRIDGE_RESEARCH_RL_TYPES_H

#include "torch/torch.h"
#include <torch/extension.h>
#include <unordered_map>

namespace rl {


using TorchJitInput = std::vector<torch::jit::IValue>;
using TorchJitOutput = torch::jit::IValue;
using TorchJitModel = torch::jit::script::Module;
using Action = int;
using Player = int;

enum PlayerId {
    // Player 0 is always valid, and is used in single-player games.
    kDefaultPlayerId = 0,
    // The fixed player id for chance/nature.
    kChancePlayerId = -1,
    // What is returned as a player id when the game is simultaneous.
    kSimultaneousPlayerId = -2,
    // Invalid player.
    kInvalidPlayer = -3,
    // What is returned as the player id on terminal nodes.
    kTerminalPlayerId = -4,
    // player id of a mean field node
    kMeanFieldPlayerId = -5
};

struct PlayerAction {
    int player;
    int action;
};


}  // namespace rl

#endif //BRIDGE_RESEARCH_RL_TYPES_H
