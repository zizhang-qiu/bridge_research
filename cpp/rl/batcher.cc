//
// Created by qzz on 2023/7/13.
//

#include "batcher.h"

namespace rl{

class FutureReply_{
  public:
  FutureReply_() = default;

  TensorDict Get(int slot) {
    std::unique_lock<std::mutex> lock(m_ready_);
    cv_ready_.wait(lock, [this] { return ready_; });
    lock.unlock();

    TensorDict ret;
    for (const auto& key_value : data_) {
      assert(slot >= 0 && slot < key_value.second.size(0));
      ret[key_value.first] = key_value.second[slot];
    }
    return ret;
  }

  void Set(TensorDict&& data) {
    {
      std::lock_guard<std::mutex> lock(m_ready_);
      ready_ = true;
      data_ = std::move(data);
    }
    cv_ready_.notify_all();
  }

 private:
  // No need for protection, only set() can set it
  TensorDict data_;

  std::mutex m_ready_;
  bool ready_{false};
  std::condition_variable cv_ready_;
};

TensorDict FutureReply::Get() {
  assert(fut_ != nullptr);
  auto ret = fut_->Get(slot);
  fut_ = nullptr;
  return ret;
}

Batcher::Batcher(int batch_size)
    : batch_size_(batch_size)
    , next_slot_(0)
    , num_active_write_(0)
    , filling_reply_(std::make_shared<FutureReply_>())
    , filled_reply_(nullptr) {
  assert(batch_size_ > 0);
}

// Send data into batcher
FutureReply Batcher::Send(const TensorDict& data) {
  std::unique_lock<std::mutex> lock(m_next_slot_);

  // Init buffer
  if (filling_buffer_.empty()) {
    assert(filled_buffer_.empty());
    filling_buffer_ = AllocateBatchStorage(data, batch_size_);
    filled_buffer_ = AllocateBatchStorage(data, batch_size_);
  } else {
    // Check they have same keys
    if (data.size() != filling_buffer_.size()) {
      std::cout << "key in buffer: " << '\n';
      utils::PrintMapKey(filling_buffer_);
      std::cout << "key in data: " << '\n';
      utils::PrintMapKey(data);
      assert(false);
    }
  }

  assert(next_slot_ <= batch_size_);
  // Wait if current batch is full and not extracted
  cv_next_slot_.wait(lock, [this] { return next_slot_ < batch_size_; });

  int slot = next_slot_;
  ++next_slot_;
  ++num_active_write_;
  lock.unlock();

  // This will copy
  for (const auto& key_value : data) {
    // For each key, check they have same sizes
    if (filling_buffer_[key_value.first][slot].sizes() != key_value.second.sizes()) {
      std::cout << "cannot batch data, batcher need size: "
                << filling_buffer_[key_value.first][slot].sizes()
                << ", get: " << key_value.second.sizes() << '\n';
    }
    filling_buffer_[key_value.first][slot] = key_value.second;
  }

  // Batch has not been extracted yet
  assert(num_active_write_ > 0);
  assert(filling_reply_ != nullptr);
  auto reply = filling_reply_;
  lock.lock();
  --num_active_write_;
  lock.unlock();
  if (num_active_write_ == 0) {
    cv_get_batch_.notify_one();
  }
  return {reply, slot};
}

// Get batch input from batcher
TensorDict Batcher::Get() {
  std::unique_lock<std::mutex> lock(m_next_slot_);
  cv_get_batch_.wait(
      lock, [this] { return (next_slot_ > 0 && num_active_write_ == 0) || exit_; });

  if (exit_) {
    return {};
  }

  int bsize = next_slot_;
  next_slot_ = 0;
  // Assert previous reply has been handled
  assert(filled_reply_ == nullptr);
  std::swap(filling_buffer_, filled_buffer_);
  std::swap(filling_reply_, filled_reply_);
  filling_reply_ = std::make_shared<FutureReply_>();

  lock.unlock();
  cv_next_slot_.notify_all();

  TensorDict batch;
  for (const auto& key_value : filled_buffer_) {
    batch[key_value.first] = key_value.second.narrow(0, 0, bsize).contiguous();
  }

  return batch;
}

// Set batch reply for batcher
void Batcher::Set(TensorDict&& data) {
  for (const auto& key_value : data) {
    assert(key_value.second.device().is_cpu());
  }
  filled_reply_->Set(std::move(data));
  filled_reply_ = nullptr;
}

}