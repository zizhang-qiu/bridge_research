//
// Created by qzz on 2023/6/13.
//
#include "logger.h"
namespace rl::utils {
Logger::Logger(const std::string &filename) {
  logfile_.open(filename, std::ios::app);
  if (!logfile_.is_open()) {
    std::cerr << "Failed to open log file: " << filename << std::endl;
  }
}
Logger::~Logger() {
  if (logfile_.is_open()) {
    logfile_.close();
  }
}
void Logger::WriteLog(const std::string &message) {
  if (logfile_.is_open()) {
    logfile_ << message << std::endl;
  }
}

}