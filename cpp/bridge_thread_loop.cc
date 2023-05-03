//
// Created by qzz on 2023/5/3.
//
#include "bridge_thread_loop.h"

namespace rl::bridge {
void VecEnvEvalThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  torch::Tensor reward, terminal;
  obs = env_ns_->Reset(obs);
  Player current_player = kNorth;
  while (!env_ns_->AllTerminated()) {
    if (current_player % 2 == 0) {
      reply = train_actor_->Act(obs);
    } else {
      reply = oppo_actor_->Act(obs);
    }
    std::tie(obs, reward, terminal) = env_ns_->Step(reply);
    current_player = (current_player + 1) % kNumPlayers;
  }

  current_player = kNorth;
  obs = {};
  obs = env_ew_->Reset(obs);
  while (!env_ew_->AllTerminated()) {
    if (current_player % 2 == 1) {
      reply = train_actor_->Act(obs);
    } else {
      reply = oppo_actor_->Act(obs);
    }
    std::tie(obs, reward, terminal) = env_ew_->Step(reply);
    current_player = (current_player + 1) % kNumPlayers;
  }
  terminated_ = true;
}

void BridgeThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  torch::Tensor reward, terminal;

  while (!Terminated()) {
    obs = env_->Reset(obs);
    while (!env_->AnyTerminated()) {
      if (Terminated()) {
        break;
      }
      if (pause_signal) {
        paused_ = true;
        WaitUntilResume();
      }
      reply = actor_->Act(obs);
      std::tie(obs, reward, terminal) = env_->Step(reply);
    }
  }
  terminated_ = true;
}

void ImpThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  torch::Tensor reward, terminal;

  while (!Terminated()) {
    obs = env_->Reset(obs);
    while (!env_->AnyTerminated()) {
      if (Terminated()) {
        break;
      }
      if (pause_signal) {
        paused_ = true;
        WaitUntilResume();
      }
      reply = actor_->Act(obs);
      std::tie(obs, reward, terminal) = env_->Step(reply);
    }
  }
  terminated_ = true;
}

}