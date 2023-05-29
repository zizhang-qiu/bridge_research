//
// Created by qzz on 2023/2/25.
//

#ifndef BRIDGE_RESEARCH_RL_LOGGING_H
#define BRIDGE_RESEARCH_RL_LOGGING_H
#include <cassert>
#include <iostream>
#include <string>
#include "str_utils.h"
namespace rl::logging{
template <class... Args>
void AssertWithMessage(bool condition, Args&&... args) {
    if (!condition) {
        const std::string msg = rl::utils::StrCat(std::forward<Args>(args)...);
        std::cerr << msg << std::endl;
        abort();
    }
}

#define RL_CHECK_TRUE(condition, ...)                                             \
  rl::logging::AssertWithMessage(condition, #condition, " check failed at ", \
                                   __FILE__, ":", __LINE__, ". ",              \
                                   ##__VA_ARGS__);


#define RL_CHECK_NOTNULL(x, ...)                                          \
  rl::logging::AssertWithMessage(                                         \
      (x) != nullptr, #x " is not nullptr check failed at ", __FILE__, ":", \
      __LINE__, ". ", ##__VA_ARGS__)

#define RL_CHECK_EQ(x, y, ...)                                              \
  rl::logging::AssertWithMessage(                                           \
      (x) == (y), #x " == " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_NE(x, y, ...)                                              \
  rl::logging::AssertWithMessage(                                           \
      (x) != (y), #x " != " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_LT(x, y, ...)                                            \
  rl::logging::AssertWithMessage(                                         \
      (x) < (y), #x " < " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_LE(x, y, ...)                                              \
  rl::logging::AssertWithMessage(                                           \
      (x) <= (y), #x " <= " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_GT(x, y, ...)                                            \
  rl::logging::AssertWithMessage(                                         \
      (x) > (y), #x " > " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_GE(x, y, ...)                                              \
  rl::logging::AssertWithMessage(                                           \
      (x) >= (y), #x " >= " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)
}
#endif //BRIDGE_RESEARCH_RL_LOGGING_H
