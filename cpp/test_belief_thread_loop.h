//
// Created by qzz on 2024/2/29.
//

#ifndef BRIDGE_RESEARCH_CPP_TEST_BELIEF_THREAD_LOOP_H_
#define BRIDGE_RESEARCH_CPP_TEST_BELIEF_THREAD_LOOP_H_

#include "rl/thread_loop.h"

namespace rl::bridge {
class TestBeliefThreadLoop : public rl::ThreadLoop {
 public:
  TestBeliefThreadLoop(const std::vector<std::vector<Action >> &trajectories,
                       const std::shared_ptr<VecEnvActor> &actor)
      : trajectories_(trajectories),
        actor_(actor) {}

  void MainLoop() override;

 private:
  std::vector<std::vector<Action>> trajectories_;
  std::shared_ptr<bridge::VecEnvActor> actor_;

};
}
#endif //BRIDGE_RESEARCH_CPP_TEST_BELIEF_THREAD_LOOP_H_
