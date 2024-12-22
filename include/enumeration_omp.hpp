#pragma once

#include <chrono>
#include <string>

#include "framework.hpp"
#include "regex.hpp"

class EnumerationMatcher final : public Matcher {
public:
  EnumerationMatcher(std::shared_ptr<const DFA> dfa, size_t thread_num)
      : Matcher(dfa), thread_num(thread_num),
        L(thread_num, std::vector<int>(dfa->size, -1)) {}

  const std::string name() override { return "OpenMP Enumeration"; }
  bool match(const ustring &text) override;
  void start_timer();
  double stop_timer();

private:
  size_t thread_num;
  std::vector<std::vector<int>> L;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
      end_time;
};