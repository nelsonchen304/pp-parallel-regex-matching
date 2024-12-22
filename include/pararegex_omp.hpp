#pragma once
#include <bitset>
#include <memory>
#include <omp.h>
#include <string>
#include <utility>

#include "framework.hpp"
#include "regex.hpp"

using namespace std;

struct MSU {
  int id;
  bitset<MAX_NFA_STATES> mapping;
};

class OmpParaMatcher final : public Matcher {
public:
  OmpParaMatcher(std::shared_ptr<const DFA> dfa, size_t num_threads);
  const std::string name() override { return "OpenMP ParaRegex"; }
  bool match(const ustring &input) override;

private:
  size_t num_threads;
  vector<vector<pair<int, int>>> dfa_col;
  void initialize_dfa_col();
  void build_MSUs();
  void init_MSU_set_local();
  vector<MSU> MSU_set[ALPHABET_SIZE];
  vector<vector<MSU>> MSU_set_local;
  int **id_to_mapping;
};
