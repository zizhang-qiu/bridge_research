//
// Created by qzz on 2023/2/28.
//

#ifndef BRIDGE_RESEARCH_RL_CONTEXT_H
#define BRIDGE_RESEARCH_RL_CONTEXT_H
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <string>
#include "thread_loop.h"
namespace rl {
class Context {
 public:
  Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  ~Context();

  int PushThreadLoop(std::shared_ptr<ThreadLoop> thread_loop);
  void Start();
  void Pause();
  bool AllPaused() const;
  void Resume();
  void Terminate();
  bool Terminated();
  int NumThreads() const;
 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<ThreadLoop>> loops_;
  std::vector<std::thread> threads_;
};

}//namespace rl


#endif //BRIDGE_RESEARCH_RL_CONTEXT_H
