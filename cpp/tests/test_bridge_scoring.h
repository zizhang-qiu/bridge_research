//
// Created by qzz on 2023/2/25.
//
#include <iostream>
#include "../bridge_scoring.h"
#include "../logging.h"
using namespace rl::bridge;
void TestScore(){
    // under tricks
    RL_CHECK_EQ(Score({3, kNoTrump, kUndoubled}, 6, false), -150);
    RL_CHECK_EQ(Score({3, kNoTrump, kDoubled}, 6, false), -500);
    RL_CHECK_EQ(Score({4, kNoTrump, kUndoubled}, 8, true), -200);
    RL_CHECK_EQ(Score({3, kNoTrump, kRedoubled}, 6, true), -1600);

    // made tricks
    RL_CHECK_EQ(Score({4, kHearts, kUndoubled}, 11, true), 650);
    RL_CHECK_EQ(Score({4, kDiamonds, kUndoubled}, 10, true), 130);

    RL_CHECK_EQ(Score({2, kSpades, kDoubled}, 8, true), 670);
    RL_CHECK_EQ(Score({3, kNoTrump, kRedoubled}, 10, false), 1000);
    RL_CHECK_EQ(Score({4, kClubs, kRedoubled}, 11, true), 1320);
    std::cout << "Passed scoring test." << std::endl;
}