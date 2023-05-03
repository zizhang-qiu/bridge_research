//
// Created by qzz on 2023/2/28.
//

#ifndef BRIDGE_RESEARCH_THREAD_LOOP_H
#define BRIDGE_RESEARCH_THREAD_LOOP_H
#include <thread>
#include <mutex>
#include <atomic>
#include <utility>
#include "bridge_actor.h"
#include "bridge_scoring.h"
#include "bridge_envs.h"
#include "replay_buffer.h"
#include "imp_env.h"
namespace rl {

class ThreadLoop {
 public:
  ThreadLoop() = default;

  ThreadLoop(const ThreadLoop &) = delete;

  ThreadLoop &operator=(const ThreadLoop &) = delete;

  virtual ~ThreadLoop() = default;
  virtual void Terminate() {terminated_ = true;}
  virtual void Pause();
  virtual void Resume();
  virtual void WaitUntilResume();
  virtual bool Terminated() const{return terminated_;}
  virtual bool Paused() const{return paused_;}
  virtual void MainLoop() = 0;

 protected:
  std::atomic_bool terminated_{false};
  std::mutex m_paused_;
  bool pause_signal = false;
  bool paused_ = false;
  std::condition_variable cvPaused_;
};

} //namespace rl
#endif //BRIDGE_RESEARCH_THREAD_LOOP_H
