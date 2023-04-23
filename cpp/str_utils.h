//
// Created by qzz on 2023/4/17.
//

#ifndef BRIDGE_RESEARCH_STR_UTILS_H
#define BRIDGE_RESEARCH_STR_UTILS_H
#include <memory>
#include <stdexcept>
#include <string>
namespace rl::utils {
template <typename... Args>
std::string StrFormat(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

template<typename T>
void StrAppendImpl(std::string* str, T arg) {
  std::ostringstream ss;
  ss << arg;
  str->append(ss.str());
}

template<typename T, typename... Args>
void StrAppendImpl(std::string* str, T arg, Args... args) {
  StrAppendImpl(str, arg);
  StrAppendImpl(str, args...);
}

template<typename... Args>
void StrAppend(std::string* str, Args... args) {
  StrAppendImpl(str, args...);
}

template <typename T>
std::string StrJoin(const T& values, const std::string& delimiter) {
  std::ostringstream os;
  auto it = std::begin(values);
  const auto end = std::end(values);
  if (it != end) {
    os << *it++;
  }
  while (it != end) {
    os << delimiter << *it++;
  }
  return os.str();
}
} // namespace rl::utils

#endif // BRIDGE_RESEARCH_STR_UTILS_H
