//
// Created by qzz on 2023/2/23.
//

#ifndef BRIDGE_RESEARCH_RL_UTILS_H
#define BRIDGE_RESEARCH_RL_UTILS_H

//#undef snprintf

#include "types.h"
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace rl::utils {

template <typename T>
inline std::vector<T> Slice(const std::vector<T>& vec, size_t start, size_t end) {
  if (end > vec.size()) {
    end = vec.size();
  }

  if (start >= end) {
    return std::vector<T>();  // Return an empty vector if start is greater than or equal to end
  }

  return std::vector<T>(vec.begin() + start, vec.begin() + end);
}

template<typename T, std::size_t N>
inline std::vector<T> GetTopKElements(const std::array<T, N> &arr, int k) {
  std::vector<T> copy(arr.begin(), arr.end());
  std::partial_sort(copy.begin(), copy.begin() + k, copy.end(), std::greater<T>());
  return std::vector<T>(copy.begin(), copy.begin() + k);
}

template<typename T>
inline std::vector<T> ConcatenateVectors(const std::vector<T> &vector1, const std::vector<T> &vector2) {
  std::vector<T> concatenated_vector;
  concatenated_vector.reserve(vector1.size() + vector2.size());

  concatenated_vector.insert(concatenated_vector.end(), vector1.begin(), vector1.end());
  concatenated_vector.insert(concatenated_vector.end(), vector2.begin(), vector2.end());

  return concatenated_vector;
}

inline bool CheckProbNotZero(const torch::Tensor &action,
                             const torch::Tensor &log_probs) {
  auto index = action.item<int>();
  auto value = torch::exp(log_probs[index]).item<float>();
  return value != 0.0;
}

inline std::tuple<double, double>
ComputeMeanAndSem(const std::vector<int> &data) {
  int n = static_cast<int>(data.size());
  double sum = 0.0;
  double sum_squared = 0.0;
  for (int i = 0; i < n; i++) {
    sum += data[i];
    sum_squared += data[i] * data[i];
  }
  double mean = sum / n;
  double variance = (sum_squared / n) - (mean * mean);
  double sem = std::sqrt(variance / n);
  return std::make_tuple(mean, sem);
}

template<typename T, std::size_t ROW, std::size_t COL>
inline void Print2DArray(const T (&arr)[ROW][COL]) {
  for (std::size_t i = 0; i < ROW; ++i) {
    for (std::size_t j = 0; j < COL; ++j) {
      std::cout << arr[i][j] << " ";
    }
    std::cout << "\n";
  }
}

template<typename T, std::size_t N>
inline void PrintArray(const std::array<T, N> &arr) {
  for (size_t i = 0; i < N; ++i) {
    std::cout << arr[i] << ", " << std::endl;
  }
}

template<typename T>
inline T SumUpVector(const std::vector<T> &vec) {
  T sum{};
  for (const auto elem : vec) {
    sum += elem;
  }
  return sum;
}

template<typename T>
inline std::vector<T> RepeatVector(const std::vector<T> &vec, int n) {
  std::vector<T> result;
  result.reserve(vec.size() * n);  // Reserve space for the repeated vector

  for (int i = 0; i < n; ++i) {
    result.insert(result.end(), vec.begin(), vec.end());
  }

  return result;
}

inline std::vector<int> VectorMod(const std::vector<int> &vec, int num) {
  std::vector<int> ret;
  ret.reserve(vec.size());
  for (auto v : vec) {
    ret.emplace_back(v % num);
  }
  return ret;
}

inline std::vector<float> VectorDiv(const std::vector<int> &vec, int num) {
  std::vector<float> ret;
  ret.reserve(vec.size());
  for (auto v : vec) {
    ret.emplace_back(v / num);
  }
  return ret;
}

template<typename T>
inline bool IsValueInVector(const std::vector<T> &v, T value) {
  auto it = std::find(v.begin(), v.end(), value);
  return (it != v.end());
}

template<typename T>
inline std::vector<T> Zeros(int size) {
  assert(size > 0);
  std::vector<T> ret(size);
  for(size_t i=0; i<size;++i){
    ret[i] = T{};
  }
  return ret;
}

inline std::vector<int> Arange(int begin, int end) {
  assert(end > begin);
  std::vector<int> ret(end - begin);
  for (int i = begin; i < end; i++) {
    ret[i - begin] = i;
  }
  return ret;
}

inline std::vector<int> Permutation(int begin, int end, std::mt19937 &rng) {
  std::vector<int> input = Arange(begin, end);
  std::shuffle(input.begin(), input.end(), rng);
  return input;
}

template<typename T, std::size_t N>
void PrintArray(const std::array<std::optional<T>, N> &arr) {
  std::cout << "[";
  for (std::size_t i = 0; i < N; ++i) {
    if (arr[i].has_value()) {
      std::cout << arr[i].value();
    } else {
      std::cout << "nullopt";
    }
    if (i < N - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

inline int GetProduct(const std::vector<int64_t> &nums) {
  int prod = 1;
  for (auto v : nums) {
    prod *= v;
  }
  return prod;
}

template<typename T>
inline std::vector<T> PushLeft(T left, const std::vector<T> &vals) {
  std::vector<T> vec;
  vec.reserve(1 + vals.size());
  vec.push_back(left);
  for (auto v : vals) {
    vec.push_back(v);
  }
  return vec;
}

template<typename T>
inline void PrintVector(const std::vector<T> &vec) {
  for (const auto &v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template<typename T>
inline void PrintMapKey(const T &map) {
  for (const auto &name2sth : map) {
    std::cout << name2sth.first << ", ";
  }
  std::cout << std::endl;
}

template<typename T>
inline void PrintMap(const T &map) {
  for (const auto &name2sth : map) {
    std::cout << name2sth.first << ": " << name2sth.second << std::endl;
  }
  // std::cout << std::endl;
}

inline torch::Tensor GetTopkActions(const torch::Tensor &policy, int k,
                                    float min_prob) {
  torch::Tensor digits, indices;
  std::tie(digits, indices) = torch::topk(policy, k);
  std::vector<int> ret;
  for (size_t i = 0; i < indices.size(0); i++) {
    if (digits[i].item<float>() > min_prob) {
      ret.emplace_back(indices[i].item<int>());
    }
  }
  return torch::tensor(torch::ArrayRef<int>(ret));
}

inline std::vector<int> GetOneHotIndices(const std::vector<float> &tensor) {
  std::vector<int> indices;
  for (int i = 0; i < tensor.size(); ++i) {
    if (tensor[i] == 1.0) {
      indices.push_back(i);
    }
  }
  return indices;
}

inline std::vector<int> GetNonZeroIndices(const std::vector<float> &tensor) {
  std::vector<int> indices;
  for (int i = 0; i < tensor.size(); ++i) {
    if (tensor[i] != 0.0) {
      indices.push_back(i);
    }
  }
  return indices;
}

inline void PrintProgressBar(int percent) {
  std::string bar;
  for (int i = 0; i < 50; i++) {
    if (i < (percent / 2)) {
      bar.replace(i, 1, "=");
    } else if (i == (percent / 2)) {
      bar.replace(i, 1, ">");
    } else {
      bar.replace(i, 1, " ");
    }
  }
  std::cout << "\r"
               "["
            << bar << "] ";
  std::cout.width(3);
  std::cout << percent << "%     " << std::flush;
}

inline void PrintProgressBar(int current, int total,
                             std::chrono::steady_clock::time_point start) {
  // Calculate progress percentage
  float progress = static_cast<float>(current) / total;

  // Get current time
  auto now = std::chrono::steady_clock::now();

  // Calculate elapsed time
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
          .count();

  // Calculate estimated time left
  int remaining = static_cast<int>((elapsed / progress) - elapsed);

  // Print progress bar
  int barWidth = 40;
  std::cout << "[";
  int pos = static_cast<int>(barWidth * progress);
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << static_cast<int>(progress * 100.0) << "% ";

  // Print estimated time left
  int hours = remaining / (1000 * 60 * 60);
  int minutes = (remaining / (1000 * 60)) % 60;
  int seconds = (remaining / 1000) % 60;
  std::cout << hours << "h " << minutes << "m " << seconds << "s remaining"
            << std::endl;
}

template<typename T>
void PrintTensor(const torch::Tensor &tensor) {
  if (tensor.dim() != 1) {
    std::cerr << "Error: Tensor must be 1D" << std::endl;
    return;
  }

  auto accessor = tensor.accessor<T, 1>();

  std::cout << "[";
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    std::cout << accessor[i];
    if (i != tensor.size(0) - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

template<typename T>
std::string FormatTensor(const torch::Tensor &tensor) {
  if (tensor.dim() != 1) {
    std::cerr << "Error: Tensor must be 1D" << std::endl;
  }

  auto accessor = tensor.accessor<T, 1>();
  std::string ret = "[";
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    ret += std::to_string(accessor[i]);
    if (i != tensor.size(0) - 1) {
      ret += ", ";
    }
  }
  ret += "]";
  return ret;
}
} // namespace rl::utils

#endif // BRIDGE_RESEARCH_RL_UTILS_H
