//
// Created by qzz on 2023/2/23.
//

#ifndef BRIDGE_RESEARCH_UTILS_H
#define BRIDGE_RESEARCH_UTILS_H

#undef snprintf

#include <iostream>
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <random>
#include "types.h"

namespace rl::utils {


inline std::tuple<double, double> ComputeMeanAndSem(const std::vector<int> &data) {
    int n = data.size();
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

template<class... Args>
inline std::string StrCat(const Args &... args) {
    using Expander = int[];
    std::stringstream ss;
    (void) Expander{0, (void(ss << args), 0)...};
    return ss.str();
}

inline std::vector<int> VectorMod(const std::vector<int> &vec, int num) {
    std::vector<int> ret;
    for (auto v: vec) {
        ret.emplace_back(v % num);
    }
    return ret;
}

inline std::vector<float> VectorDiv(const std::vector<int> &vec, int num) {
    std::vector<float> ret;
    for (auto v: vec) {
        ret.emplace_back(v / num);
    }
    return ret;
}

template<typename T>
inline bool IsValueInVector(const std::vector<T> &v, T value) {
    auto it = std::find(v.begin(), v.end(), value);
    return (it != v.end());
}

inline std::vector<int> Arange(int begin, int end) {
    assert(end > begin);
    std::vector<int> ret(end - begin);
    for (int i = begin; i < end; i++) {
        ret[i - begin] = i;
    }
    return ret;
}

inline std::vector<int> ArangeShuffle(int begin, int end, std::mt19937 &rng) {
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
    for (auto v: nums) {
        prod *= v;
    }
    return prod;
}

template<typename T>
inline std::vector<T> PushLeft(T left, const std::vector<T> &vals) {
    std::vector<T> vec;
    vec.reserve(1 + vals.size());
    vec.push_back(left);
    for (auto v: vals) {
        vec.push_back(v);
    }
    return vec;
}


template<typename T>
inline void PrintVector(const std::vector<T> &vec) {
    for (const auto &v: vec) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
}

template<typename T>
inline void PrintMapKey(const T &map) {
    for (const auto &name2sth: map) {
        std::cout << name2sth.first << ", ";
    }
    std::cout << std::endl;
}

template<typename T>
inline void PrintMap(const T &map) {
    for (const auto &name2sth: map) {
        std::cout << name2sth.first << ": " << name2sth.second << std::endl;
    }
    // std::cout << std::endl;
}


inline torch::Tensor GetTopkActions(const torch::Tensor &policy, int k, float min_prob) {
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
    std::cout << "\r" "[" << bar << "] ";
    std::cout.width(3);
    std::cout << percent << "%     " << std::flush;
}

}  // namespace rl:utils

#endif //BRIDGE_RESEARCH_UTILS_H
