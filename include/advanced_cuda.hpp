#pragma once
#include <memory>
#include <omp.h>
#include <string>

#include "framework.hpp"
#include "regex.hpp"

using namespace std;

class AdvancedCudaMatcher final : public Matcher {
public:
  AdvancedCudaMatcher(std::shared_ptr<const DFA> dfa, bool use_omp);
  const std::string name() override {
    std::string ret = "Advanced CUDA";
    if (use_omp) {
      ret += " + OpenMP";
    }
    return ret;
  }
  bool match(const ustring &input) override;

private:
  size_t num_threads;
  bool use_omp;

  void extract_mapping();
  void extract_dfa_transitions();
  void extract_active_states();
  int *dfa_trans;
  int *mapping;
  vector<int> active_states[ALPHABET_SIZE];
  int active_states_limit;
};
