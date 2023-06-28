//
// Created by qzz on 2023/6/13.
//

#ifndef BRIDGE_RESEARCH_CPP_RL_LOGGER_H_
#define BRIDGE_RESEARCH_CPP_RL_LOGGER_H_
#include <fstream>
#include <iostream>
#include <string>

namespace rl::utils{
class Logger {
public:
  explicit Logger(const std::string& filename);
  ~Logger();
  void WriteLog(const std::string& message);

private:
  std::ofstream logfile_;
};
}
#endif //BRIDGE_RESEARCH_CPP_RL_LOGGER_H_
