//
// Created by qzz on 2023/6/26.
//

#ifndef BRIDGE_RESEARCH_CPP_SCORE_PREDICTOR_H_
#define BRIDGE_RESEARCH_CPP_SCORE_PREDICTOR_H_
#include <utility>

#include "model_locker.h"
#include "tensor_dict.h"
namespace rl::bridge {
class ScorePredictor {
 public:
  ScorePredictor(std::shared_ptr<ModelLocker> model_locker)
      : model_locker_(std::move(model_locker)) {}

  TensorDict Predict(const TensorDict &final_obs) {
    torch::NoGradGuard ng;
    TorchJitInput input;
    int id = -1;
    auto model = model_locker_->GetModel(&id);
    input.emplace_back(rl::tensor_dict::ToTorchDict(final_obs, model_locker_->device_));
    auto output = model.get_method("predict")(input);
    auto reply = rl::tensor_dict::FromIValue(output, torch::kCPU, true);
    model_locker_->ReleaseModel(id);
    return reply;
  }
 private:
  std::shared_ptr<ModelLocker> model_locker_;
};
}
#endif //BRIDGE_RESEARCH_CPP_SCORE_PREDICTOR_H_
