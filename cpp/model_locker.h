//
// Created by qzz on 2023/2/23.
//

#ifndef BRIDGE_RESEARCH_MODEL_LOCKER_H
#define BRIDGE_RESEARCH_MODEL_LOCKER_H
#include <vector>
#include <string>
#include <torch/torch.h>
#include "types.h"
#include "utils.h"
namespace rl {
class ModelLocker {
public:
    ModelLocker(std::vector<py::object> py_models, const std::string &device)
            : device_(torch::Device(device)), py_models_(py_models), model_call_counts_(py_models.size(), 0),
              latest_model_(0) {
        // assert(py_models_.Size() > 1);
        for (size_t i = 0; i < py_models_.size(); ++i) {
            models_.push_back(py_models_[i].attr("_c").cast<TorchJitModel *>());
            // model_call_counts_.push_back(0);
        }
    }

    ModelLocker(std::vector<TorchJitModel*> models, const std::string &device):
            device_(torch::Device(device)), model_call_counts_(models.size(), 0),
            latest_model_(0){
        for (size_t i = 0; i < models.size(); ++i) {
            models_.push_back(models[i]);
        }
    }

    void UpdateModel(py::object pyModel) {
        std::unique_lock<std::mutex> lk(m_);
        int id = (latest_model_ + 1) % model_call_counts_.size();
//    std::cout << id << std::endl;
//    std::cout << model_call_counts_ << std::endl;
        cv_.wait(lk, [this, id] { return model_call_counts_[id] == 0; });
        lk.unlock();

        py_models_[id].attr("load_state_dict")(pyModel.attr("state_dict")());

        lk.lock();
        latest_model_ = id;
        lk.unlock();
    }

    const TorchJitModel GetModel(int *id) {
        std::lock_guard<std::mutex> lk(m_);
        *id = latest_model_;
        // std::cout << "using mdoel: " << latest_model_ << std::endl;
        ++model_call_counts_[latest_model_];
        return *models_[latest_model_];
    }

    void ReleaseModel(int id) {
        std::unique_lock<std::mutex> lk(m_);
        --model_call_counts_[id];
        if (model_call_counts_[id] == 0) {
            cv_.notify_one();
        }
    }

    torch::Device GetDevice() const {
        return device_;
    }

    const torch::Device device_;

private:
    // py::function model_cons_;
    std::vector<py::object> py_models_;
    std::vector<int> model_call_counts_;
    int latest_model_;

    std::vector<TorchJitModel *> models_;
    std::mutex m_;
    std::condition_variable cv_;
};
}
#endif //BRIDGE_RESEARCH_MODEL_LOCKER_H
