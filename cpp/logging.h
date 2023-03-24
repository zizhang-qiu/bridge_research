//
// Created by qzz on 2023/2/25.
//
#include <cassert>
#include <iostream>
#include <string>
#include "utils.h"
#ifndef BRIDGE_RESEARCH_LOGGING_H
#define BRIDGE_RESEARCH_LOGGING_H
namespace rl::logging{
template <class... Args>
void assertWithMessage(bool condition, Args&&... args) {
    if (!condition) {
        const std::string msg = rl::utils::StrCat(std::forward<Args>(args)...);
        std::cerr << msg << std::endl;
        abort();
    }
}

#define RL_CHECK_TRUE(condition, ...)                                             \
  rl::logging::assertWithMessage(condition, #condition, " check failed at ", \
                                   __FILE__, ":", __LINE__, ". ",              \
                                   ##__VA_ARGS__);


#define RL_CHECK_NOTNULL(x, ...)                                          \
  rl::logging::assertWithMessage(                                         \
      (x) != nullptr, #x " is not nullptr check failed at ", __FILE__, ":", \
      __LINE__, ". ", ##__VA_ARGS__)

#define RL_CHECK_EQ(x, y, ...)                                              \
  rl::logging::assertWithMessage(                                           \
      (x) == (y), #x " == " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_NE(x, y, ...)                                              \
  rl::logging::assertWithMessage(                                           \
      (x) != (y), #x " != " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_LT(x, y, ...)                                            \
  rl::logging::assertWithMessage(                                         \
      (x) < (y), #x " < " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_LE(x, y, ...)                                              \
  rl::logging::assertWithMessage(                                           \
      (x) <= (y), #x " <= " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_GT(x, y, ...)                                            \
  rl::logging::assertWithMessage(                                         \
      (x) > (y), #x " > " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)

#define RL_CHECK_GE(x, y, ...)                                              \
  rl::logging::assertWithMessage(                                           \
      (x) >= (y), #x " >= " #y, " check failed at ", __FILE__, ":", __LINE__, \
      ": ", (x), " vs ", (y), ". ", ##__VA_ARGS__)
}
#endif //BRIDGE_RESEARCH_LOGGING_H
