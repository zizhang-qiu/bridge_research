//
// Created by qzz on 2023/7/13.
//

#ifndef BRIDGE_RESEARCH_CPP_RL_BATCHER_H_
#define BRIDGE_RESEARCH_CPP_RL_BATCHER_H_
#include "rl/utils.h"
#include "rl/tensor_dict.h"
namespace rl{

inline TensorDict AllocateBatchStorage(const TensorDict& data, int size) {
  TensorDict storage;
  for (const auto& key_value : data) {
    auto t = key_value.second.sizes();
    std::vector<int64_t> sizes;
    sizes.push_back(size);
    for (long long i : t) {
      sizes.push_back(i);
    }

    storage[key_value.first] = torch::zeros(sizes, key_value.second.dtype());
  }
  return storage;
}

class FutureReply_;

class FutureReply {
 public:
  FutureReply()
      : fut_(nullptr)
      , slot(-1) {
  }

  FutureReply(std::shared_ptr<FutureReply_> fut, int slot)
      : fut_(std::move(fut))
      , slot(slot) {
  }

  TensorDict Get();

  [[nodiscard]] bool IsNull() const {
    return fut_ == nullptr;
  }

 private:
  std::shared_ptr<FutureReply_> fut_;
  int slot;
};

using Future = FutureReply;

class Batcher {
 public:
  explicit Batcher(int batch_size);

  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

  ~Batcher() {
    Exit();
  }

  void Exit() {
    {
      std::unique_lock<std::mutex> lk(m_next_slot_);
      exit_ = true;
    }
    cv_get_batch_.notify_all();
  }

  [[nodiscard]] bool Terminated() const {
    return exit_;
  }

  // Send data into batcher
  FutureReply Send(const TensorDict& t);

  // Get batch input from batcher
  TensorDict Get();

  // Set batch reply for batcher
  void Set(TensorDict&& t);

 private:
  const int batch_size_;

  int next_slot_;
  int num_active_write_;
  std::condition_variable cv_next_slot_;

  TensorDict filling_buffer_;
  std::shared_ptr<FutureReply_> filling_reply_;

  TensorDict filled_buffer_;
  std::shared_ptr<FutureReply_> filled_reply_;

  bool exit_ = false;
  std::condition_variable cv_get_batch_;
  std::mutex m_next_slot_;
};

}
#endif //BRIDGE_RESEARCH_CPP_RL_BATCHER_H_
