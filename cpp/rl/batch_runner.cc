//
// Created by qzz on 2023/7/13.
//

#include "batch_runner.h"

namespace rl{

FutureReply BatchRunner::Call(const std::string& method, const TensorDict& data) const {
  auto batcher_it = batchers_.find(method);
  if (batcher_it == batchers_.end()) {
    std::cerr << "Error: Cannot find method: " << method << '\n';
    std::cerr << "available methods are:" << '\n';
    for (auto& key_value : batchers_) {
      std::cerr << key_value.first << '\n';
    }
    assert(false);
  }
  return batcher_it->second->Send(data);
}

void BatchRunner::Start() {
  // Create a batcher for each method.
  for (size_t i = 0; i < methods_.size(); ++i) {
    batchers_.emplace(methods_[i], std::make_unique<Batcher>(batch_sizes_[i]));
  }

  // Create threads for each batcher.
  for (auto& key_value : batchers_) {
    threads_.emplace_back(&BatchRunner::RunnerLoop, this, key_value.first);
  }
}

void BatchRunner::Stop() {
  // batchers_.clear();
  for (auto& key_value : batchers_) {
    key_value.second->Exit();
  }

  for (auto& thr : threads_) {
    if (thr.joinable()) {
      thr.join();
    }
  }
}

// For debugging
rl::TensorDict BatchRunner::BlockCall(const std::string& method, const TensorDict& data) {
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> input;
  input.push_back(tensor_dict::ToIValue(data, device_));
  torch::jit::IValue output;
  {
    std::lock_guard<std::mutex> lock(mtx_update_);
    output = jit_model_->get_method(method)(input);
  }
  return tensor_dict::FromIValue(output, torch::kCPU, true);
}

void BatchRunner::RunnerLoop(const std::string& method) {
  // Find the method.
  auto batcher_it = batchers_.find(method);
  if (batcher_it == batchers_.end()) {
    std::cerr << "Error: RunnerLoop, Cannot find method: " << method << '\n';
    assert(false);
  }
  auto& batcher = *(batcher_it->second);

  // Aggregate size and count
  int agg_size = 0;
  int agg_count = 0;

  while (!batcher.Terminated()) {
    auto batch = batcher.Get();
    if (batch.empty()) {
      assert(batcher.Terminated());
      break;
    }

    if (log_freq_ > 0) {
      agg_size += static_cast<int>(batch.begin()->second.size(0));
      agg_count += 1;

      if (agg_count % log_freq_ == 0) {
        std::cout << method << ", average batch_size: " << static_cast<float>(agg_size) / static_cast<float>(agg_count)
                  << ", call count: " << agg_count << '\n';
        agg_size = 0;
        agg_count = 0;
      }
    }

    {
      std::lock_guard<std::mutex> lock(mtx_device_);

      torch::NoGradGuard no_grad;
      std::vector<torch::jit::IValue> input;
      input.push_back(tensor_dict::ToIValue(batch, device_));
      torch::jit::IValue output;
      {
        std::lock_guard<std::mutex> lk2(mtx_update_);
        output = jit_model_->get_method(method)(input);
      }
      batcher.Set(tensor_dict::FromIValue(output, torch::kCPU, true));
    }
  }
}

}