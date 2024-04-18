//
// Created by qzz on 2023/7/7.
//

#ifndef BRIDGE_RESEARCH_CPP_GAME_PARAMETERS_H_
#define BRIDGE_RESEARCH_CPP_GAME_PARAMETERS_H_
#include <unordered_map>
#include <string>
namespace rl{
using Params = std::unordered_map<std::string, std::string>;

// Returns string associated with key in params, parsed as template type.
// If key is not in params, returns the provided default value.
template <class T>
T ParameterValue(const Params& params,
                 const std::string& key, T default_value);

template <>
int ParameterValue(const Params& params,
                   const std::string& key, int default_value);
template <>
double ParameterValue(
    const Params& params,
    const std::string& key, double default_value);
template <>
std::string ParameterValue(
    const Params& params,
    const std::string& key, std::string default_value);
template <>
bool ParameterValue(const Params& params,
                    const std::string& key, bool default_value);
}
#endif //BRIDGE_RESEARCH_CPP_GAME_PARAMETERS_H_
