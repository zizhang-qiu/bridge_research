//
// Created by qzz on 2023/2/23.
//

#ifndef BRIDGE_RESEARCH_RL_MODEL_LOCKER_H
#define BRIDGE_RESEARCH_RL_MODEL_LOCKER_H
#include <vector>
#include <string>
#include <torch/torch.h>
#include "types.h"
#include "utils.h"
namespace rl {
class ModelLocker {
 public:
  ModelLocker(std::vector<py::object> py_models, const std::string &device);

  ModelLocker(std::vector<TorchJitModel *> models, const std::string &device);

  void UpdateModel(py::object py_model);

  const TorchJitModel GetModel(int *id);

  void ReleaseModel(int id);

  std::string GetDevice() const { return device_.str(); }

  const torch::Device device_;

 private:
  std::vector<py::object> py_models_;
  std::vector<int> model_call_counts_;
  int latest_model_;

  std::vector<TorchJitModel *> models_;
  std::mutex m_;
  std::condition_variable cv_;
};
}
#endif //BRIDGE_RESEARCH_RL_MODEL_LOCKER_H
