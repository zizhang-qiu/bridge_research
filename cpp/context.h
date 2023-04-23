//
// Created by qzz on 2023/2/28.
//

#ifndef BRIDGE_RESEARCH_CONTEXT_H
#define BRIDGE_RESEARCH_CONTEXT_H
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <string>
#include "thread_loop.h"
namespace rl{
class Context {
public:
    Context()
            :started_(false)
            ,numTerminatedThread_(0){};

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context(){
        for (auto& v : loops_) {
            v->Terminate();
        }
        for (auto& v : threads_) {
            v.join();
        }
    }

    int PushThreadLoop(std::shared_ptr<ThreadLoop> thread_loop) {
        assert(!started_);
        loops_.push_back(std::move(thread_loop));
        return (int)loops_.size();
    }

    void Start() {
        for (int i = 0; i < (int)loops_.size(); ++i) {
//            loops_[i]->SetThreadID(i);
//            std::cout << "thread" << i << "starting" << std::endl;
            threads_.emplace_back([this, i]() {
                loops_[i]->MainLoop();
                ++numTerminatedThread_;
            });
//            std::cout << "thread" << i << "started" << std::endl;
        }
    }

    void Pause() {
        for (auto& v : loops_) {
            v->Pause();
        }
    }

    bool AllPaused() {
        for (auto& v : loops_) {
            if(!v->Paused()){
                return false;
            }
        }
        return true;
    }

    void Resume() {
        for (auto& v : loops_) {
            v->Resume();
        }
    }

    void Terminate() {
        for (auto& v : loops_) {
            v->Terminate();
        }
    }

    bool Terminated() {
        // std::cout << ">>> " << numTerminatedThread_ << std::endl;
        for (auto& v:loops_) {
            if(!v->Terminated()){
                return false;
            }
        }
        return true;
//        return numTerminatedThread_ == (int)loops_.size();
    }

    int NumThreads(){
        return loops_.size();
    }
private:
    bool started_;
    std::atomic<int> numTerminatedThread_;
    std::vector<std::shared_ptr<ThreadLoop>> loops_;
    std::vector<std::thread> threads_;
};

}//namespace rl


#endif //BRIDGE_RESEARCH_CONTEXT_H
