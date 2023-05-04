//
// Created by qzz on 2023/5/4.
//
#include "model_locker.h"
namespace rl {
ModelLocker::ModelLocker(std::vector<py::object> py_models, const std::string &device)
    : device_(torch::Device(device)), py_models_(py_models), model_call_counts_(py_models.size(), 0),
      latest_model_(0) {
  for (size_t i = 0; i < py_models_.size(); ++i) {
    models_.push_back(py_models_[i].attr("_c").cast<TorchJitModel *>());
  }
}

ModelLocker::ModelLocker(std::vector<TorchJitModel *> models, const std::string &device) :
    device_(torch::Device(device)), model_call_counts_(models.size(), 0),
    latest_model_(0) {
  for (size_t i = 0; i < models.size(); ++i) {
    models_.push_back(models[i]);
  }
}

void ModelLocker::UpdateModel(py::object py_model) {
  std::unique_lock<std::mutex> lk(m_);
  int id = (latest_model_ + 1) % model_call_counts_.size();
  cv_.wait(lk, [this, id] { return model_call_counts_[id] == 0; });
  lk.unlock();

  py_models_[id].attr("load_state_dict")(py_model.attr("state_dict")());

  lk.lock();
  latest_model_ = id;
  lk.unlock();
}

const TorchJitModel ModelLocker::GetModel(int *id) {
  std::lock_guard<std::mutex> lk(m_);
  *id = latest_model_;
  // std::cout << "using mdoel: " << latest_model_ << std::endl;
  ++model_call_counts_[latest_model_];
  return *models_[latest_model_];
}

void ModelLocker::ReleaseModel(int id) {
  std::unique_lock<std::mutex> lk(m_);
  --model_call_counts_[id];
  if (model_call_counts_[id] == 0) {
    cv_.notify_one();
  }
}

}
