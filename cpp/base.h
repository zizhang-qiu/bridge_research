//
// Created by qzz on 2023/3/4.
//
#include "torch/torch.h"

#ifndef BRIDGE_RESEARCH_BASE_H
#define BRIDGE_RESEARCH_BASE_H
namespace rl {
class Env {
public:
    Env() = default;

    ~Env() = default;

    virtual torch::Tensor Reset() = 0;

    virtual std::tuple<torch::Tensor, float, bool> Step(const torch::Tensor &action) = 0;

    virtual bool Terminated() const = 0;
};

class Actor {
public:
    Actor() = default;

    ~Actor() = default;

    virtual torch::Tensor Act(const torch::Tensor &obs) = 0;
};
}
#endif //BRIDGE_RESEARCH_BASE_H
