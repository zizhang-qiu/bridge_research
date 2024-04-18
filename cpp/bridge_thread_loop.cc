//
// Created by qzz on 2023/5/3.
//
#include "bridge_thread_loop.h"
#include "bridge_lib/dds_call.h"

namespace rl::bridge {
void VecEnvEvalThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  env_ns_->Reset();
  obs = env_ns_->GetFeature();
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
    obs = env_ns_->GetFeature();
//    std::cout << "got feature." << std::endl;
    current_player = (current_player + 1) % kNumPlayers;
  }
//  std::cout << "env ns end." << std::endl;

  current_player = kNorth;
  obs = {};
  env_ew_->Reset();
  obs = env_ew_->GetFeature();
  while (!env_ew_->AllTerminated()) {
//    std::cout << current_player << std::endl;
    if (current_player % 2 == 1) {
      reply = train_actor_->Act(obs);
    } else {
      reply = oppo_actor_->Act(obs);
    }
    env_ew_->Step(reply);
    obs = env_ew_->GetFeature();
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

void BridgeVecEnvThreadLoop::MainLoop() {
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

void VecEnvAllTerminateThreadLoop::MainLoop() {
  TensorDict obs = {};
  TensorDict reply;
  env_->Reset();
  while (!env_->AllTerminated()) {
    obs = env_->GetFeature();
    reply = actor_->Act(obs);
    env_->Step(reply);
  }

  terminated_ = true;
}

void BeliefThreadLoop::MainLoop() {
  TensorDict obs;
  TensorDict reply;
  TensorDict belief;
  while (!Terminated()) {
    if (Terminated() || replay_->Size()==replay_->Capacity()) {
      break;
    }
    if (pause_signal) {
      paused_ = true;
      WaitUntilResume();
    }
    env_->Reset();
    while (!env_->AnyTerminated()) {
      obs = env_->GetFeature();
      belief = env_->GetBeliefFeature();
      auto transition = ObsBelief(obs, belief);
      transition.length = torch::tensor(env_->BiddingLengths());
      replay_->Add(transition, torch::ones(env_->Size()));
      reply = actor_->Act(obs);
      env_->Step(reply);
    }
  }
  terminated_ = true;
}


void ContractScoreThreadLoop::MainLoop() {
  int contract_index;
  BridgeDeal deal;

  std::shared_ptr<BridgeBiddingState> state;
  std::vector<FinalObsScore> obs_scores;
  while (!Terminated()) {
    if (Terminated()) {
      break;
    }

    state = std::make_shared<BridgeBiddingState>(deal);
    deal = deal_manager_->Next();
    for(int i=0; i < batch_size_; ++i) {
      contract_index = dis_(rng_);
      auto final_observation_tensor = state->FinalObservationTensor(contract_index);
      auto score = state->ScoreForContracts(kNorth, {contract_index})[0];
      TensorDict final_obs = {
          {"final_s", torch::tensor(final_observation_tensor)}
      };
      torch::Tensor score_tensor = torch::tensor(score, {torch::kFloat32});
      FinalObsScore f_obs_score(final_obs, score_tensor);
      obs_scores.push_back(f_obs_score);
    }
    replay_->Add(obs_scores, torch::ones(batch_size_, {torch::kFloat32}));
    obs_scores.clear();
  }
  terminated_ = true;
}
}