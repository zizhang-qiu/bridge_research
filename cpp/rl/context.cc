//
// Created by qzz on 2023/5/3.
//
#include "context.h"

namespace rl {

Context::Context(): started_(false), numTerminatedThread_(0) {}

Context::~Context(){
  for (auto &v : loops_) {
    v->Terminate();
  }
  for (auto &v : threads_) {
    v.join();
  }
}

int Context::PushThreadLoop(std::shared_ptr<ThreadLoop> thread_loop) {
  assert(!started_);
  loops_.push_back(std::move(thread_loop));
  return static_cast<int>(loops_.size());
}

void Context::Start() {
  for (int i = 0; i < (int) loops_.size(); ++i) {
    threads_.emplace_back([this, i]() {
      loops_[i]->MainLoop();
      ++numTerminatedThread_;
    });
  }
}

void Context::Pause() {
  for (auto &v : loops_) {
    v->Pause();
  }
}

bool Context::AllPaused() const {
  for (auto &v : loops_) {
    if (!v->Paused()) {
      return false;
    }
  }
  return true;
}

void Context::Resume() {
  for (auto &v : loops_) {
    v->Resume();
  }
}

void Context::Terminate() {
  for (auto &v : loops_) {
    v->Terminate();
  }
}

bool Context::Terminated() {
  for (auto &v : loops_) {
    if (!v->Terminated()) {
      return false;
    }
  }
  return true;
}

int Context::NumThreads() const {
  return static_cast<int>(loops_.size());
}

}