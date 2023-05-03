//
// Created by qzz on 2023/5/3.
//

#ifndef BRIDGE_RESEARCH_CPP_BRIDGE_THREAD_LOOP_H_
#define BRIDGE_RESEARCH_CPP_BRIDGE_THREAD_LOOP_H_
#include "rl/thread_loop.h"
namespace rl::bridge {
class VecEnvEvalThreadLoop : public ThreadLoop {
 public:
  VecEnvEvalThreadLoop(std::shared_ptr<bridge::VecEnvActor> train_actor,
                       std::shared_ptr<bridge::VecEnvActor> oppo_actor,
                       std::shared_ptr<BridgeVecEnv> env_ns,
                       std::shared_ptr<BridgeVecEnv> env_ew) :
      train_actor_(std::move(train_actor)),
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
                   std::shared_ptr<VecEnvActor> actor) :
      env_(std::move(env)),
      actor_(std::move(actor)) {}

  void MainLoop() override;

 private:
  std::shared_ptr<BridgeVecEnv> env_;
  std::shared_ptr<VecEnvActor> actor_;
};

class ImpThreadLoop : public ThreadLoop {
 public:
  ImpThreadLoop(std::shared_ptr<ImpVecEnv> env,
                std::shared_ptr<VecEnvActor> actor) :
      env_(std::move(env)), actor_(std::move(actor)) {}

  void MainLoop() override;
 private:
  std::shared_ptr<ImpVecEnv> env_;
  std::shared_ptr<VecEnvActor> actor_;
};
}
#endif //BRIDGE_RESEARCH_CPP_BRIDGE_THREAD_LOOP_H_