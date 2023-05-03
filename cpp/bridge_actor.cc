//
// Created by qzz on 2023/5/3.
//
#include "bridge_actor.h"
namespace rl::bridge {

TensorDict SingleEnvActor::Act(const TensorDict &obs) {
  torch::NoGradGuard ng;
  TorchJitInput input;
  int id = -1;
  auto model = model_locker_->GetModel(&id);
  input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
  auto output = model.get_method("act")(input);
  auto reply = tensor_dict::FromIValue(output, torch::kCPU, true);

  model_locker_->ReleaseModel(id);
  return reply;
}

double SingleEnvActor::GetProbForAction(const TensorDict &obs, Action action) {
  torch::NoGradGuard ng;
  TorchJitInput input;
  int id = -1;
  auto model = model_locker_->GetModel(&id);
  input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
  input.emplace_back(action);
  auto output = model.get_method("get_prob_for_action")(input);
  return output.toDouble();
}

TensorDict VecEnvActor::Act(const TensorDict &obs) {
  torch::NoGradGuard ng;
  TorchJitInput input;
  int id = -1;
  auto model = model_locker_->GetModel(&id);
  input.emplace_back(tensor_dict::ToTorchDict(obs, model_locker_->device_));
  auto output = model.get_method("act")(input);
  auto reply = tensor_dict::FromIValue(output, torch::kCPU, true);
  model_locker_->ReleaseModel(id);
  return reply;
}

} // namespace rl::bridge