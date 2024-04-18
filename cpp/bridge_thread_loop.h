//
// Created by qzz on 2023/5/3.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_THREAD_LOOP_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_THREAD_LOOP_H_

#include <utility>
#include "rl/thread_loop.h"

namespace rl::bridge {
class VecEnvEvalThreadLoop : public ThreadLoop {
 public:
  VecEnvEvalThreadLoop(std::shared_ptr<bridge::VecEnvActor> train_actor,
                       std::shared_ptr<bridge::VecEnvActor> oppo_actor,
                       std::shared_ptr<BridgeVecEnv> env_ns,
                       std::shared_ptr<BridgeVecEnv> env_ew)
      : train_actor_(std::move(train_actor)),
        oppo_actor_(std::move(oppo_actor)),
        env_ns_(std::move(env_ns)),
        env_ew_(std::move(env_ew)) {
  };

  void MainLoop() override;

 private:
  std::shared_ptr<bridge::VecEnvActor> train_actor_;
  std::shared_ptr<bridge::VecEnvActor> oppo_actor_;
  std::shared_ptr<BridgeVecEnv> env_ns_;
  std::shared_ptr<BridgeVecEnv> env_ew_;
};

class BridgeThreadLoop : public ThreadLoop {
 public:
  BridgeThreadLoop(std::shared_ptr<BridgeVecEnv> env,
                   std::shared_ptr<VecEnvActor> actor)
      : env_(std::move(env)),
        actor_(std::move(actor)) {}

  void MainLoop() override;

 private:
  std::shared_ptr<BridgeVecEnv> env_;
  std::shared_ptr<VecEnvActor> actor_;
};

class ImpThreadLoop : public ThreadLoop {
 public:
  ImpThreadLoop(std::shared_ptr<ImpVecEnv> env,
                std::shared_ptr<VecEnvActor> actor)
      : env_(std::move(env)),
        actor_(std::move(actor)) {}

  void MainLoop() override;

 private:
  std::shared_ptr<ImpVecEnv> env_;
  std::shared_ptr<VecEnvActor> actor_;
};

class BridgeVecEnvThreadLoop : public ThreadLoop {
 public:
  BridgeVecEnvThreadLoop(std::shared_ptr<BridgeWrapperVecEnv> env,
                         std::shared_ptr<VecEnvActor> actor)
      : env_(std::move(env)),
        actor_(std::move(actor)) {}

  void MainLoop() override;

 private:
  std::shared_ptr<BridgeWrapperVecEnv> env_;
  std::shared_ptr<VecEnvActor> actor_;
};

class VecEnvAllTerminateThreadLoop : public ThreadLoop {
 public:
  VecEnvAllTerminateThreadLoop(std::shared_ptr<VecEnvActor> actor,
                               std::shared_ptr<BridgeVecEnv> env)
      : actor_(std::move(actor)),
        env_(std::move(env)) {}

  void MainLoop() override;

 private:
  std::shared_ptr<VecEnvActor> actor_;
  std::shared_ptr<BridgeVecEnv> env_;
};

class BeliefThreadLoop : public ThreadLoop {
 public:
  BeliefThreadLoop(std::shared_ptr<VecEnvActor> actor,
                   std::shared_ptr<BridgeVecEnv> env,
                   std::shared_ptr<ObsBeliefReplay> replay)
      : actor_(std::move(actor)),
        env_(std::move(env)),
        replay_(std::move(replay)) {}

  void MainLoop() override;

 private:
  std::shared_ptr<VecEnvActor> actor_;
  std::shared_ptr<BridgeVecEnv> env_;
  std::shared_ptr<ObsBeliefReplay> replay_;
};

class ContractScoreThreadLoop : public ThreadLoop {
 public:
  ContractScoreThreadLoop(std::shared_ptr<BridgeDealManager> deal_manager,
                          std::shared_ptr<FinalObsScoreReplay> replay,
                          int batch_size,
                          int seed)
      : deal_manager_(std::move(deal_manager)),
        replay_(std::move(replay)),
        batch_size_(batch_size),
        rng_(seed) {
    dis_ = std::uniform_int_distribution<int>(1, kNumContracts - 1);
  }

  void MainLoop() override;

 private:
  std::shared_ptr<BridgeDealManager> deal_manager_;
  std::mt19937 rng_;
  std::uniform_int_distribution<int> dis_;
  int batch_size_;
  std::shared_ptr<FinalObsScoreReplay> replay_;
};
}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_THREAD_LOOP_H_
