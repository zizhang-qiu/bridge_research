//
// Created by qzz on 2023/7/7.
//
#include "game_parameters.h"
namespace rl{
template <>
int ParameterValue<int>(
    const Params& params,
    const std::string& key, int default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stoi(iter->second);
}

template <>
std::string ParameterValue<std::string>(
    const Params& params,
    const std::string& key, std::string default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return iter->second;
}

template <>
double ParameterValue<double>(
    const Params& params,
    const std::string& key, double default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stod(iter->second);
}

template <>
bool ParameterValue<bool>(
    const Params& params,
    const std::string& key, bool default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return iter->second == "1" || iter->second == "true" ||
      iter->second == "True";
}
}