//
// Created by qzz on 2023/5/3.
//
#include "bridge_thread_loop.h"

namespace rl::bridge {
void VecEnvEvalThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  env_ns_->Reset();
  obs = env_ns_->GetFeatures();
  Player current_player = kNorth;
  while (!env_ns_->AllTerminated()) {
//    std::cout << current_player << std::endl;
    if (current_player % 2 == 0) {
      reply = train_actor_->Act(obs);
    } else {
      reply = oppo_actor_->Act(obs);
    }
    env_ns_->Step(reply);
//    std::cout << "stepped." << std::endl;
    obs = env_ns_->GetFeatures();
//    std::cout << "got feature." << std::endl;
    current_player = (current_player + 1) % kNumPlayers;
  }
//  std::cout << "env ns end." << std::endl;

  current_player = kNorth;
  obs = {};
  env_ew_->Reset();
  obs = env_ew_->GetFeatures();
  while (!env_ew_->AllTerminated()) {
//    std::cout << current_player << std::endl;
    if (current_player % 2 == 1) {
      reply = train_actor_->Act(obs);
    } else {
      reply = oppo_actor_->Act(obs);
    }
    env_ew_->Step(reply);
    obs = env_ew_->GetFeatures();
    current_player = (current_player + 1) % kNumPlayers;
  }
  terminated_ = true;
}

void BridgeThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  torch::Tensor reward, terminal;

  while (!Terminated()) {
    env_->Reset();
    obs = env_->GetFeatures();
    while (!env_->AnyTerminated()) {
      if (Terminated()) {
        break;
      }
      if (pause_signal) {
        paused_ = true;
        WaitUntilResume();
      }
      reply = actor_->Act(obs);
      env_->Step(reply);
      obs = env_->GetFeatures();
    }
  }
  terminated_ = true;
}

void ImpThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;

  while (!Terminated()) {
    env_->Reset();
    obs = env_->GetFeature();
    while (!env_->AnyTerminated()) {
      if (Terminated()) {
        break;
      }
      if (pause_signal) {
        paused_ = true;
        WaitUntilResume();
      }
      reply = actor_->Act(obs);
      env_->Step(reply);
      obs = env_->GetFeature();
    }
  }
  terminated_ = true;
}

}