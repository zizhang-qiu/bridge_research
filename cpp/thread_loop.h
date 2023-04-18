//
// Created by qzz on 2023/2/28.
//
#include <thread>
#include <mutex>
#include <atomic>
#include <utility>
#include "bridge_env.h"
#include "bridge_actor.h"
#include "bridge_scoring.h"
#include "replay_buffer.h"

#ifndef BRIDGE_RESEARCH_THREAD_LOOP_H
#define BRIDGE_RESEARCH_THREAD_LOOP_H
namespace rl {

template<typename T>
class ConcurrentVector {
public:
    ConcurrentVector() = default;

    void PushBack(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        vec_.push_back(value);
    }

    void PushBackNoWait(T value){
        vec_.push_back(value);
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return vec_.size();
    }

    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return vec_.empty();
    }

    std::vector<T> GetVector() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return vec_;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        vec_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::vector<T> vec_;
};

using IntConVec = ConcurrentVector<int>;


class ThreadLoop {
public:
    ThreadLoop() = default;

    ThreadLoop(const ThreadLoop &) = delete;

    ThreadLoop &operator=(const ThreadLoop &) = delete;

    virtual ~ThreadLoop() = default;

    virtual void Terminate() {
        terminated_ = true;
    }

    virtual void Pause() {
        std::lock_guard<std::mutex> lk(mPaused_);
        pause_signal = true;
    }

    virtual void Resume() {
        {
            std::lock_guard<std::mutex> lk(mPaused_);
            pause_signal = false;
        }
        cvPaused_.notify_one();
    }

    virtual void WaitUntilResume() {
        std::unique_lock<std::mutex> lk(mPaused_);
        cvPaused_.wait(lk, [this] { return !pause_signal; });
    }

    virtual bool Terminated() {
        return terminated_;
    }

    virtual bool Paused() {
        return paused_;
    }

    virtual void SetThreadID(int id) {
        thread_id = id;
    }

    virtual void MainLoop() = 0;

private:

protected:
    int thread_id = 0;
    std::atomic_bool terminated_{false};
    std::mutex mPaused_;
    bool pause_signal = false;
    bool paused_ = false;
    std::condition_variable cvPaused_;
};

class EvalThreadLoop : public ThreadLoop {
public:
    EvalThreadLoop(std::shared_ptr<bridge::BridgeBiddingEnv> env_0,
                   std::shared_ptr<bridge::BridgeBiddingEnv> env_1,
                   std::shared_ptr<bridge::SingleEnvActor> actor_train,
                   std::shared_ptr<bridge::SingleEnvActor> actor_oppo_,
                   int num_deals,
                   std::shared_ptr<IntConVec> imp_vec,
                   bool verbose
    ) : env_0_(std::move(env_0)),
        env_1_(std::move(env_1)),
        actor_train_(std::move(actor_train)),
        actor_oppo_(std::move(actor_oppo_)),
        imp_vec_(std::move(imp_vec)),
        num_deals_(num_deals),
        verbose_(verbose) {};

    void MainLoop() override {
        torch::Tensor obs, action;
        float r;
        bool t;
        Player current_player;
        double score_0, score_1;
        for (int i = 0; i < num_deals_; i++) {
            obs = env_0_->Reset();
            while (!env_0_->Terminated()) {
                current_player = env_0_->CurrentPlayer();
                if (current_player % 2 == 0) {
                    action = actor_train_->Act(obs);
                } else {
                    action = actor_oppo_->Act(obs);
                }
                std::tie(obs, r, t) = env_0_->Step(action);
            }
            score_0 = env_0_->Returns()[bridge::kNorth];
            if (verbose_) {
                std::cout << "env 0:\n" << env_0_->ToString() << std::endl;
            }

            obs = env_1_->Reset();
            while (!env_1_->Terminated()) {
                current_player = env_1_->CurrentPlayer();
                if (current_player % 2 == 1) {
                    action = actor_train_->Act(obs);
                } else {
                    action = actor_oppo_->Act(obs);
                }
                std::tie(obs, r, t) = env_1_->Step(action);
            }
            score_1 = env_1_->Returns()[bridge::kNorth];
            if (verbose_) {
                std::cout << "env 1:\n" << env_1_->ToString() << std::endl;
            }
            const int imp = bridge::GetImp(int(score_0), int(score_1));
            if (verbose_) {
                std::cout << "imp: " << imp << std::endl;
            }
            imp_vec_->PushBack(imp);
        }
        terminated_ = true;
    };
private:
    std::shared_ptr<bridge::BridgeBiddingEnv> env_0_;
    std::shared_ptr<bridge::BridgeBiddingEnv> env_1_;
    std::shared_ptr<bridge::SingleEnvActor> actor_train_;
    std::shared_ptr<bridge::SingleEnvActor> actor_oppo_;
    std::shared_ptr<IntConVec> imp_vec_;
    bool verbose_;
    int num_deals_;
};

class VecEvalThreadLoop : public ThreadLoop {
public:
    VecEvalThreadLoop(std::shared_ptr<bridge::BridgeVecEnv> vec_env_0,
                      std::shared_ptr<bridge::BridgeVecEnv> vec_env_1,
                      std::shared_ptr<bridge::VecEnvActor> actor_train,
                      std::shared_ptr<bridge::VecEnvActor> actor_oppo,
                      std::shared_ptr<IntConVec> imp_vec,
                      bool verbose,
                      int num_loops = 1) :
            vec_env_0_(std::move(vec_env_0)),
            vec_env_1_(std::move(vec_env_1)),
            actor_train_(std::move(actor_train)),
            actor_oppo_(std::move(actor_oppo)),
            imp_vec_(std::move(imp_vec)),
            verbose_(verbose),
            num_loops_(num_loops) {};

    void MainLoop() override {
        torch::Tensor obs, r, t, action;
        Player current_player;
        for (int i_loop = 0; i_loop < num_loops_; i_loop++) {
            if (verbose_) std::cout << "enter loop" << std::endl;
            obs = vec_env_0_->Reset();
            current_player = bridge::kNorth;
            while (!vec_env_0_->AllTerminated()) {
                if (current_player % 2 == 0) {
                    action = actor_train_->Act(obs);
                } else {
                    action = actor_oppo_->Act(obs);
                }
                std::tie(obs, r, t) = vec_env_0_->Step(action);
                current_player = (current_player + 1) % bridge::kNumPlayers;
            }
            if (verbose_) {
                vec_env_0_->Display(3);
            }
            obs = vec_env_1_->Reset();
            current_player = bridge::kNorth;
            while (!vec_env_1_->AllTerminated()) {
                if (current_player % 2 == 1) {
                    action = actor_train_->Act(obs);
                } else {
                    action = actor_oppo_->Act(obs);
                }
                std::tie(obs, r, t) = vec_env_1_->Step(action);
                current_player = (current_player + 1) % bridge::kNumPlayers;
            }
            if (verbose_) {
                vec_env_1_->Display(3);
            }
            auto scores_0 = vec_env_0_->Returns(bridge::kNorth);
            auto scores_1 = vec_env_1_->Returns(bridge::kNorth);
            for (int i = 0; i < scores_0.size(); i++) {
                auto imp = bridge::GetImp(int(scores_0[i]), int(scores_1[i]));
                imp_vec_->PushBack(imp);
            }
        }
        terminated_ = true;
    }

private:
    std::shared_ptr<bridge::BridgeVecEnv> vec_env_0_;
    std::shared_ptr<bridge::BridgeVecEnv> vec_env_1_;
    std::shared_ptr<bridge::VecEnvActor> actor_train_;
    std::shared_ptr<bridge::VecEnvActor> actor_oppo_;
    std::shared_ptr<IntConVec> imp_vec_;
    bool verbose_;
    int num_loops_;
};

class BridgePGThreadLoop : public ThreadLoop {
public:
    BridgePGThreadLoop(std::vector<bridge::SingleEnvActor> actors,
                       std::shared_ptr<bridge::BridgeBiddingEnv> env_0,
                       std::shared_ptr<bridge::BridgeBiddingEnv> env_1,
                       std::shared_ptr<bridge::ReplayBuffer> buffer,
                       bool verbose) {
        RL_CHECK_EQ(actors.size(), bridge::kNumPlayers);
        actors_ = std::move(actors);
        env_0_ = std::move(env_0);
        env_1_ = std::move(env_1);
        buffer_ = std::move(buffer);
        verbose_ = verbose;
    };

    void MainLoop() override {
        torch::Tensor obs, action;
        float r;
        bool t;
        Player current_player;
        int acting_player;
        double score_0, score_1;
        while (!Terminated()) {
            try {
                obs = env_0_->Reset();
                while (!env_0_->Terminated()) {
                    if (Terminated()) {
                        break;
                    }
                    if (pause_signal) {
                        paused_ = true;
                        WaitUntilResume();
                    }
                    current_player = env_0_->CurrentPlayer();
                    acting_player = current_player;
                    action = actors_[acting_player].Act(obs);
                    std::tie(obs, r, t) = env_0_->Step(action);
                    actors_[acting_player].SetRewardAndTerminal(r, t);
                }
                if (Terminated()) {
                    break;
                }
                if (pause_signal) {
                    paused_ = true;
                    WaitUntilResume();
                }
                score_0 = env_0_->Returns()[bridge::kNorth];
                if (verbose_) {
                    std::cout << "env 0:\n" << env_0_->ToString() << std::endl;
                }

                obs = env_1_->Reset();
                while (!env_1_->Terminated()) {
                    if (Terminated()) {
                        break;
                    }
                    if (pause_signal) {
                        paused_ = true;
                        WaitUntilResume();
                    }
                    current_player = env_1_->CurrentPlayer();
                    acting_player = (current_player + 1) % bridge::kNumPlayers;
                    action = actors_[acting_player].Act(obs);
                    std::tie(obs, r, t) = env_1_->Step(action);
                    actors_[acting_player].SetRewardAndTerminal(r, t);
                }
                if (Terminated()) {
                    break;
                }
                if (pause_signal) {
                    paused_ = true;
                    WaitUntilResume();
                }
                score_1 = env_1_->Returns()[bridge::kNorth];
                if (verbose_) {
                    std::cout << "env 1:\n" << env_1_->ToString() << std::endl;
                }
                int imp = bridge::GetImp(int(score_0), int(score_1));
                if (verbose_) {
                    std::cout << "imp: " << imp << std::endl;
                }
                auto reward = (float) imp / bridge::kMaxImp;
                for (int i = 0; i < bridge::kNumPlayers; i += 2) {
                    actors_[i].PostToReplayBuffer(buffer_, i % 2 == 0 ? reward : -reward);
                }
            } catch (std::runtime_error &e) {
                std::cout << e.what() << std::endl;
            }
        }
    };


private:
    std::vector<bridge::SingleEnvActor> actors_;
    std::shared_ptr<bridge::BridgeBiddingEnv> env_0_;
    std::shared_ptr<bridge::BridgeBiddingEnv> env_1_;
    std::shared_ptr<bridge::ReplayBuffer> buffer_;
    bool verbose_;
};

class ImpEnvThreadLoop : public ThreadLoop {
public:
    ImpEnvThreadLoop(std::vector<std::shared_ptr<bridge::SingleEnvActor>> actors,
                     std::shared_ptr<bridge::ImpEnv> env,
                     std::shared_ptr<bridge::ReplayBuffer> buffer,
                     bool verbose) : actors_(std::move(actors)),
                                     env_(std::move(env)),
                                     buffer_(std::move(buffer)),
                                     verbose_(verbose) {
        RL_CHECK_EQ(actors_.size(), bridge::kNumPlayers);
    };

    void MainLoop() override {
        torch::Tensor obs, action;
        float r;
        bool t;
        int acting_player;
        while (!Terminated()) {
            obs = env_->Reset();
            while (!env_->Terminated()) {
                if (pause_signal) {
                    paused_ = true;
                    WaitUntilResume();
                }
                if (Terminated()) {
                    break;
                }
                acting_player = env_->ActingPlayer();
                if (verbose_) {
                    std::cout << "acting player: " << acting_player << std::endl;
                }
                action = actors_[acting_player]->Act(obs);
                std::tie(obs, r, t) = env_->Step(action);
            }
            if (pause_signal) {
                paused_ = true;
                WaitUntilResume();
            }
            if (Terminated()) {
                break;
            }
            if (verbose_) {
                std::cout << env_->ToString() << std::endl;
            }
            int imp_reward = env_->Returns()[0];
            float reward = float(imp_reward) / bridge::kMaxImp;
            if (verbose_) {
                std::cout << reward << std::endl;
            }
            for (int i = 0; i < bridge::kNumPlayers; i += 2) {
                actors_[i]->PostToReplayBuffer(buffer_, reward);
            }

        }
        terminated_ = true;
    }

private:
    std::vector<std::shared_ptr<bridge::SingleEnvActor>> actors_;
    std::shared_ptr<bridge::ImpEnv> env_;
    std::shared_ptr<bridge::ReplayBuffer> buffer_;
    bool verbose_;
};


class EvalImpThreadLoop : public ThreadLoop {
public:
    EvalImpThreadLoop(std::vector<std::shared_ptr<bridge::SingleEnvActor>> actors,
                      std::shared_ptr<bridge::ImpEnv> env,
                      const int num_deals) : actors_(std::move(actors)),
                                             env_(std::move(env)),
                                             num_deals_(num_deals) {
        RL_CHECK_EQ(actors_.size(), bridge::kNumPlayers);
    }

    void MainLoop() override {
        torch::Tensor obs, action;
        float r;
        bool t;
        int acting_player;
        for (size_t i = 0; i < num_deals_; i++) {
            obs = env_->Reset();
            while (!env_->Terminated()) {
                acting_player = env_->ActingPlayer();
                action = actors_[acting_player]->Act(obs);
                std::tie(obs, r, t) = env_->Step(action);
            }
            int imp = env_->Returns()[0];
        }
        terminated_ = true;
    }

private:
    std::vector<std::shared_ptr<bridge::SingleEnvActor>> actors_;
    std::shared_ptr<bridge::ImpEnv> env_;
    const int num_deals_;
};

} //namespace rl
#endif //BRIDGE_RESEARCH_THREAD_LOOP_H
