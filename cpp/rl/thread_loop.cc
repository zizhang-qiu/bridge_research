//
// Created by qzz on 2023/5/3.
//
#include "thread_loop.h"

namespace rl {
void ThreadLoop::Pause() {
  std::lock_guard<std::mutex> lk(m_paused_);
  pause_signal = true;
}

void ThreadLoop::Resume() {
  {
    std::lock_guard<std::mutex> lk(m_paused_);
    pause_signal = false;
  }
  cvPaused_.notify_one();
}

void ThreadLoop::WaitUntilResume(){
  std::unique_lock<std::mutex> lk(m_paused_);
  cvPaused_.wait(lk, [this] { return !pause_signal; });
}
}