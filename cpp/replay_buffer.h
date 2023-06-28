//
// Created by qzz on 2023/2/28.
//

#ifndef BRIDGE_RESEARCH_REPLAY_BUFFER_H
#define BRIDGE_RESEARCH_REPLAY_BUFFER_H
#include <atomic>
#include <future>
#include <random>
#include "torch/torch.h"
#include "transition.h"
namespace rl::bridge {
class ReplayBuffer {
 public:
  ReplayBuffer(int state_size,
               int num_actions,
               int capacity,
               float alpha,
               float eps,
               float beta);

  void Push(const torch::Tensor &state, const torch::Tensor &action, const torch::Tensor &reward,
            const torch::Tensor &log_probs);

  int Size() {
    std::unique_lock<std::mutex> lk(m_);
    if (full_) {
      return capacity_;
    }
    return int(cursor_);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  Sample(int batch_size, const std::string &device);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GetAll() {
    std::unique_lock<std::mutex> lk(m_);
    int size = full_ ? capacity_ : int(cursor_);
    auto states = state_storage_.slice(0, 0, size);
    auto actions = action_storage_.slice(0, 0, size);
    auto rewards = reward_storage_.slice(0, 0, size);
    auto log_probs = log_probs_storage_.slice(0, 0, size);
    return std::make_tuple(states, actions, rewards, log_probs);
  }

  int NumAdd() {
    return num_add_.load();
  }

 private:
  const int state_size_;
  const int num_actions_;
  const int capacity_;
  const float alpha_;
  const float eps_;
  const float beta_;
  std::atomic<int> cursor_;
  std::atomic<bool> full_;
  std::atomic<int> num_add_;
  torch::Tensor state_storage_;
  torch::Tensor action_storage_;
  torch::Tensor reward_storage_;
  torch::Tensor log_probs_storage_;
  mutable std::mutex m_;
};

template<class DataType>
class ConcurrentQueue {
 public:
  explicit ConcurrentQueue(int capacity)
      : capacity(capacity),
        head_(0),
        tail_(0),
        size_(0),
        safe_tail_(0),
        safe_size_(0),
        sum_(0),
        evicted_(capacity, false),
        elements_(capacity),
        weights_(capacity, 0) {
  }

  int SafeSize(float *sum) const {
    std::unique_lock<std::mutex> lk(m_);
    if (sum != nullptr) {
      *sum = sum_;
    }
    return safe_size_;
  }

  int Size() const {
    std::unique_lock<std::mutex> lk(m_);
    return size_;
  }

  void BlockAppend(const std::vector<DataType> &block, const torch::Tensor &weights) {
    int block_size = block.size();

    std::unique_lock<std::mutex> lk(m_);
    cv_size_.wait(lk, [=] { return size_ + block_size <= capacity; });

    int start = tail_;
    int end = (tail_ + block_size) % capacity;

    tail_ = end;
    size_ += block_size;
    CheckSize(head_, tail_, size_);

    lk.unlock();

    float sum = 0;
    auto weight_acc = weights.accessor<float, 1>();
    assert(weight_acc.size(0) == block_size);
    for (int i = 0; i < block_size; ++i) {
      int j = (start + i) % capacity;
      elements_[j] = block[i];
      weights_[j] = weight_acc[i];
      sum += weight_acc[i];
    }

    lk.lock();

    cv_tail_.wait(lk, [=] { return safe_tail_ == start; });
    safe_tail_ = end;
    safe_size_ += block_size;
    sum_ += sum;
    CheckSize(head_, safe_tail_, safe_size_);

    lk.unlock();
    cv_tail_.notify_all();
  }

  // ------------------------------------------------------------- //
  // blockPop, update are thread-safe against blockAppend
  // but they are NOT thread-safe against each other

  void BlockPop(int blockSize) {
    double diff = 0;
    int head = head_;
    for (int i = 0; i < blockSize; ++i) {
      diff -= weights_[head];
      evicted_[head] = true;
      head = (head + 1) % capacity;
    }

    {
      std::lock_guard<std::mutex> lk(m_);
      sum_ += diff;
      head_ = head;
      safe_size_ -= blockSize;
      size_ -= blockSize;
      assert(safe_size_ >= 0);
      CheckSize(head_, safe_tail_, safe_size_);
    }
    cv_size_.notify_all();
  }

  void Update(const std::vector<int> &ids, const torch::Tensor &weights) {
    double diff = 0;
    auto weight_acc = weights.accessor<float, 1>();
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
      int id = ids[i];
      if (evicted_[id]) {
        continue;
      }
      diff += (weight_acc[i] - weights_[id]);
      weights_[id] = weight_acc[i];
    }

    std::lock_guard<std::mutex> lk_(m_);
    sum_ += diff;
  }

  // ------------------------------------------------------------- //
  // accessing elements is never locked, operate safely!

  DataType Get(int idx) {
    int id = (head_ + idx) % capacity;
    return elements_[id];
  }

  DataType GetElementAndMark(int idx) {
    int id = (head_ + idx) % capacity;
    evicted_[id] = false;
    return elements_[id];
  }

  float GetWeight(int idx, int *id) {
    assert(id != nullptr);
    *id = (head_ + idx) % capacity;
    return weights_[*id];
  }

  const int capacity;

 private:
  void CheckSize(int head, int tail, int size) {
    if (size == 0) {
      assert(tail == head);
    } else if (tail > head) {
      if (tail - head != size) {
        std::cout << "tail-head: " << tail - head << " vs size: " << size << std::endl;
      }
      assert(tail - head == size);
    } else {
      if (tail + capacity - head != size) {
        std::cout << "tail-head: " << tail + capacity - head << " vs size: " << size
                  << std::endl;
      }
      assert(tail + capacity - head == size);
    }
  }

  mutable std::mutex m_;
  std::condition_variable cv_size_;
  std::condition_variable cv_tail_;

  int head_;
  int tail_;
  int size_;

  int safe_tail_;
  int safe_size_;
  double sum_;
  std::vector<bool> evicted_;

  std::vector<DataType> elements_;
  std::vector<float> weights_;
};

template<class DataType>
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta, int prefetch)
      : alpha_(alpha)  // priority exponent
      , beta_(beta)    // importance sampling exponent
      , prefetch_(prefetch), capacity_(capacity), storage_(static_cast<int>(1.25 * capacity)), num_add_(0) {
    rng_.seed(seed);
  }

  void Add(const std::vector<DataType> &sample, const torch::Tensor &priority) {
    assert(priority.dim() == 1);
    auto weights = torch::pow(priority, alpha_);
    storage_.BlockAppend(sample, weights);
    num_add_ += static_cast<int>(priority.size(0));
  }

  void Add(const DataType &sample, const torch::Tensor &priority) {
    std::vector<DataType> vec;
    int n = static_cast<int>(priority.size(0));
    for (int i = 0; i < n; ++i) {
      vec.push_back(sample.Index(i));
    }
    Add(vec, priority);
  }

  std::tuple<DataType, torch::Tensor> Sample(int batch_size, const std::string &device) {
    if (!sampled_ids_.empty()) {
      std::cout << "Error: previous samples' priority has not been updated." << std::endl;
      assert(false);
    }

    DataType batch;
    torch::Tensor priority;
    if (prefetch_ == 0) {
      std::tie(batch, priority, sampled_ids_) = Sample_(batch_size, device);
      return std::make_tuple(batch, priority);
    }

    if (futures_.empty()) {
      std::tie(batch, priority, sampled_ids_) = Sample_(batch_size, device);
    } else {
      // assert(futures_.size() == 1);
      std::tie(batch, priority, sampled_ids_) = futures_.front().get();
      futures_.pop();
    }

    while (static_cast<int>(futures_.size()) < prefetch_) {
      auto f = std::async(
          std::launch::async,
          &PrioritizedReplay<DataType>::Sample_,
          this,
          batch_size,
          device);
      futures_.push(std::move(f));
    }

    return std::make_tuple(batch, priority);
  }

  void UpdatePriority(const torch::Tensor &priority) {
    if (priority.size(0) == 0) {
      sampled_ids_.clear();
      return;
    }

    assert(priority.dim() == 1);
    assert((int) sampled_ids_.size() == priority.size(0));

    auto weights = torch::pow(priority, alpha_);
    {
      std::lock_guard<std::mutex> lk(m_sampler_);
      storage_.Update(sampled_ids_, weights);
    }
    sampled_ids_.clear();
  }

  DataType Get(int idx) {
    return storage_.Get(idx);
  }

  int Size() const {
    return storage_.SafeSize(nullptr);
  }

  int NumAdd() const {
    return num_add_;
  }

 private:
  using SampleWeightIds = std::tuple<DataType, torch::Tensor, std::vector<int>>;

  SampleWeightIds Sample_(int batch_size, const std::string &device) {
    std::unique_lock<std::mutex> lk(m_sampler_);

    float sum;
    int size = storage_.SafeSize(&sum);
    assert(size >= batch_size);
    // std::cout << "size: "<< size << ", sum: " << sum << std::endl;
    // storage_ [0, size) remains static in the subsequent section

    float segment = sum / static_cast<float>(batch_size);
    std::uniform_real_distribution<float> dist(0.0, segment);

    std::vector<DataType> samples;
    auto weights = torch::zeros({batch_size}, torch::kFloat32);
    auto weightAcc = weights.accessor<float, 1>();
    std::vector<int> ids(batch_size);

    double accSum = 0;
    int nextIdx = 0;
    float w = 0;
    int id = 0;
    for (int i = 0; i < batch_size; i++) {
      float rand = dist(rng_) + static_cast<float>(i) * segment;
      rand = std::min(sum - (float) 0.1, rand);
      // std::cout << "looking for " << i << "th/" << batch_size << " sample" <<
      // std::endl;
      // std::cout << "\ttarget: " << rand << std::endl;

      while (nextIdx <= size) {
        if (accSum > 0 && accSum >= rand) {
          assert(nextIdx >= 1);
          // std::cout << "\tfound: " << nextIdx - 1 << ", " << id << ", " <<
          // accSum << std::endl;
          DataType element = storage_.GetElementAndMark(nextIdx - 1);
          samples.push_back(element);
          weightAcc[i] = w;
          ids[i] = id;
          break;
        }

        if (nextIdx == size) {
          std::cout << "nextIdx: " << nextIdx << "/" << size << std::endl;
          std::cout << std::setprecision(10) << "accSum: " << accSum << ", sum: " << sum
                    << ", rand: " << rand << std::endl;
          assert(false);
        }

        w = storage_.GetWeight(nextIdx, &id);
        accSum += w;
        ++nextIdx;
      }
    }
    assert(static_cast<int>(samples.size()) == batch_size);

    // pop storage if full
    size = storage_.Size();
    if (size > capacity_) {
      storage_.BlockPop(size - capacity_);
    }

    // safe to unlock, because <samples> contains copies
    lk.unlock();

    weights = weights / sum;
    weights = torch::pow(size * weights, -beta_);
    weights /= weights.max();
    if (device != "cpu") {
      weights = weights.to(torch::Device(device));
    }
    auto batch = DataType::MakeBatch(samples, device);
    return std::make_tuple(batch, weights, ids);
  }

  const float alpha_;
  const float beta_;
  const int prefetch_;
  const int capacity_;

  ConcurrentQueue<DataType> storage_;
  std::atomic<int> num_add_;

  // make sure that sample & update does not overlap
  std::mutex m_sampler_;
  std::vector<int> sampled_ids_;
  std::queue<std::future<SampleWeightIds>> futures_;

  std::mt19937 rng_;
};

using Replay = PrioritizedReplay<Transition>;
using PVReplay = PrioritizedReplay<SearchTransition>;
using ObsBeliefReplay = PrioritizedReplay<ObsBelief>;
}
#endif //BRIDGE_RESEARCH_REPLAY_BUFFER_H
