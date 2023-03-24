//
// Created by qzz on 2023/3/7.
//
#include "../bridge_env.h"
#include "cards_and_ddts.h"

#ifndef BRIDGE_RESEARCH_TEST_IMP_ENV_H
#define BRIDGE_RESEARCH_TEST_IMP_ENV_H
using namespace rl::bridge;

void TestImpEnv() {
    auto greedy = std::vector<int>{0, 0, 0, 0};
    auto imp_env = std::make_shared<ImpEnv>(cards, ddts, greedy, false);
    std::cout << imp_env->Terminated() << std::endl;
    auto obs = imp_env->Reset();
    std::vector<rl::Action> bids = {3, 0, 0, 0};
    for(const auto a:bids){
        std::cout << imp_env->ActingPlayer() << std::endl;
        imp_env->Step(torch::tensor(a));
    }
    std::cout << imp_env->ToString() << std::endl;
    bids[0] = 5;
    for(const auto a:bids){
        std::cout << imp_env->ActingPlayer() << std::endl;
        imp_env->Step(torch::tensor(a));
    }
    std::cout << imp_env->ToString() << std::endl;
    std::cout << imp_env->Terminated() << std::endl;
    std::cout << imp_env->Returns() << std::endl;
}

#endif //BRIDGE_RESEARCH_TEST_IMP_ENV_H
