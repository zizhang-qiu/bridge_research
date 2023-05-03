//
// Created by qzz on 2023/2/23.
//


#ifndef BRIDGE_RESEARCH_BRIDGE_ACTOR_H
#define BRIDGE_RESEARCH_BRIDGE_ACTOR_H
#include "rl/base.h"
#include "rl/logging.h"
#include "rl/model_locker.h"
#include "replay_buffer.h"
#include "rl/tensor_dict.h"
#include <utility>
using namespace torch::indexing;
namespace rl::bridge {

class SingleEnvActor : public Actor {

 public:
  explicit SingleEnvActor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(std::move(model_locker))) {
  };

  TensorDict Act(const TensorDict &obs) override;

  double GetProbForAction(const TensorDict &obs, Action action);

 private:
  std::shared_ptr<ModelLocker> model_locker_;
};

class VecEnvActor : public Actor {
 public:
  explicit VecEnvActor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)) {};

  TensorDict Act(const TensorDict &obs) override;

 private:
  std::shared_ptr<ModelLocker> model_locker_;
};
} // namespace rl::bridge
#endif // BRIDGE_RESEARCH_BRIDGE_ACTOR_H
