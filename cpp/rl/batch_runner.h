//
// Created by qzz on 2023/7/13.
//

#ifndef BRIDGE_RESEARCH_CPP_RL_BATCH_RUNNER_H_
#define BRIDGE_RESEARCH_CPP_RL_BATCH_RUNNER_H_
#include <cassert>
#include <thread>
#include "rl/batcher.h"
#include "rl/tensor_dict.h"
namespace rl{
class BatchRunner {
 public:
  BatchRunner(
      py::object py_model,
      const std::string& device,
      int max_batch_size,
      const std::vector<std::string>& methods)
      : py_model_(py_model)
      , jit_model_(py_model_.attr("_c").cast<torch::jit::script::Module*>())
      , device_(torch::Device(device))
      , batch_sizes_(methods.size(), max_batch_size)
      , methods_(methods) {
  }

  BatchRunner(py::object py_model, const std::string& device)
      : py_model_(py_model)
      , jit_model_(py_model_.attr("_c").cast<torch::jit::script::Module*>())
      , device_(torch::Device(device)) {
  }

  BatchRunner(const BatchRunner&) = delete;
  BatchRunner& operator=(const BatchRunner&) = delete;

  ~BatchRunner() {
    Stop();
  }

  void SetLogFreq(int log_freq) {
    log_freq_ = log_freq;
  }

  void AddMethod(const std::string& method, int batchSize) {
    batch_sizes_.push_back(batchSize);
    methods_.push_back(method);
  }

  FutureReply Call(const std::string& method, const TensorDict& t) const;

  void Start();

  void Stop();

  void UpdateModel(py::object agent) {
    std::lock_guard<std::mutex> lock(mtx_update_);
    py_model_.attr("load_state_dict")(agent.attr("state_dict")());
  }

  const torch::jit::script::Module& JitModel() {
    return *jit_model_;
  }

  // For debugging
  rl::TensorDict BlockCall(const std::string& method, const TensorDict& t);

 private:
  void RunnerLoop(const std::string& method);

  py::object py_model_;
  torch::jit::script::Module* jit_model_;
  torch::Device device_;
  std::vector<int> batch_sizes_;
  std::vector<std::string> methods_;

  // Ideally this mutex should be 1 per device, thus global
  std::mutex mtx_device_;
  std::mutex mtx_update_;

  mutable std::map<std::string, std::unique_ptr<Batcher>> batchers_;
  std::vector<std::thread> threads_;

  int log_freq_ = -1;
};
}
#endif //BRIDGE_RESEARCH_CPP_RL_BATCH_RUNNER_H_
