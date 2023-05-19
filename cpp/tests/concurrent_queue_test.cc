//
// Created by qzz on 2023/5/9.
//
#include "replay_buffer.h"
#include "gtest/gtest.h"

using namespace rl::bridge;

TEST(ConcurrentQueueTest, SingleThreadTest) {
  ConcurrentQueue<int> q(200);

  std::vector<int> block = {1, 2, 3};
  torch::Tensor weight = torch::ones({3}, torch::kFloat32);
  int num_blocks = 50;

  for (int i = 0; i < num_blocks; ++i) {
    q.BlockAppend(block, weight);
  }


  // test size and sum of weights
  {
    float sum;
    int size = q.SafeSize(&sum);
    EXPECT_EQ(size, num_blocks * static_cast<int>(block.size()));
    EXPECT_EQ(sum, num_blocks * weight.sum().item<float>());
  }

  {
    int pop_size = 40;
    q.BlockPop(pop_size);
    float sum;
    int size = q.SafeSize(&sum);
    EXPECT_EQ(pop_size + size, num_blocks * static_cast<int>(block.size()));
    EXPECT_EQ(pop_size + sum, num_blocks * weight.sum().item<float>());
  }

}

TEST(ConcurrentQueueTest, MultiApeendTest) {
  int block_size = 10;
  int num_blocks = 5000;
  ConcurrentQueue<int> q(num_blocks * block_size);
  std::vector<int> block;
  block.reserve(block_size);
  for (int i = 0; i < block_size; ++i) {
    block.push_back(i);
  }
  auto weight = torch::ones({block_size}, torch::kFloat32);
  std::vector<std::future<void>> futures;
  for (int i = 0; i < num_blocks; ++i) {
    auto f = std::async(
        std::launch::async, &ConcurrentQueue<int>::BlockAppend, &q, block, weight);
    futures.push_back(std::move(f));
  }

  for (int i = 0; i < num_blocks; ++i) {
    futures[i].get();
  }

  {
    float sum;
    int size = q.SafeSize(&sum);
    EXPECT_EQ(size, num_blocks * static_cast<int>(block.size()));
    EXPECT_EQ(sum, num_blocks * weight.sum().item<float>());
  }
}

TEST(ConcurrentQueueTest, MultiApeendPopTest) {
  int block_size = 10;
  int num_blocks = 5000;
  ConcurrentQueue<int> q(num_blocks * block_size / 2);

  std::vector<int> block;
  block.reserve(block_size);
  for (int i = 0; i < block_size; ++i) {
    block.push_back(i);
  }
  auto weight = torch::ones({block_size}, torch::kFloat32);

  std::vector<std::future<void>> futures;
  for (int i = 0; i < num_blocks; ++i) {
    // std::cout << "block " << i << std::endl;
    auto f1 = std::async(
        std::launch::async, &ConcurrentQueue<int>::BlockAppend, &q, block, weight);
    futures.push_back(std::move(f1));
  }
  int k = 0;
  while (k < num_blocks) {
    while (q.SafeSize(nullptr) < block_size) {
    }
    q.BlockPop(block_size);
    ++k;
  }

  for (int i = 0; i < num_blocks; ++i) {
    futures[i].get();
  }

  {
    float sum;
    int size = q.SafeSize(&sum);
    EXPECT_EQ(size, 0);
    EXPECT_EQ(sum, 0);
    EXPECT_EQ(q.Size(), 0);
  }

}