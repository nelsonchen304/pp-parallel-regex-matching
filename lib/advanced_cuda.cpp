#include <bitset>
#include <ctime>
#include <iostream>
#include <omp.h>

#include "advanced_cuda.hpp"
#include "advanced_cuda_kernel.hpp"

using namespace std;

AdvancedCudaMatcher::AdvancedCudaMatcher(std::shared_ptr<const DFA> dfa,
                                         bool use_omp)
    : Matcher(dfa), num_threads(196600 / dfa->size), use_omp(use_omp) {
  extract_dfa_transitions();
  extract_mapping(); // for cuda enum init
  if (use_omp) {
    extract_active_states(); // for omp + cuda init
  }
}

void AdvancedCudaMatcher::extract_mapping() {
  mapping = new int[num_threads * dfa->size]();
  for (size_t i = 0; i < num_threads; ++i) {
    for (int j = 0; j < dfa->size; ++j) {
      mapping[i * dfa->size + j] = j;
    }
  }
}
void AdvancedCudaMatcher::extract_dfa_transitions() {
  dfa_trans = new int[dfa->size * ALPHABET_SIZE]();
  for (int i = 0; i < dfa->size; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      dfa_trans[i * ALPHABET_SIZE + j] = dfa->transitions[i][j];
    }
  }
}

void AdvancedCudaMatcher::extract_active_states() {
  bitset<MAX_NFA_STATES> tmp;
  int MAX_ctr = 0;

  for (int i = 0; i < ALPHABET_SIZE; ++i) {
    int ctr = 0;
    tmp.reset();
    for (int j = 0; j < dfa->size; ++j) {
      if (tmp.test(dfa->transitions[j][i]) == false) {
        active_states[i].push_back(dfa->transitions[j][i]);
        tmp.set(dfa->transitions[j][i]);
        ctr++;
      }
    }
    MAX_ctr = max(MAX_ctr, ctr);
  }
  active_states_limit = MAX_ctr;
  if (MAX_ctr > 0) {
    num_threads = (196600 / MAX_ctr);
  } else {
    std::cerr << "MAX_ctr is 0\n";
    exit(1);
  }
  mapping = new int[num_threads * active_states_limit]();
}

bool AdvancedCudaMatcher::match(const ustring &input) {
  if (use_omp) {
    int n = input.size();
    int chunk_size = n / num_threads;
#pragma omp parallel num_threads(4)
    {
#pragma omp for
      for (size_t i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        for (size_t idx = 0; idx < active_states[input[start - 1]].size();
             idx++) {
          mapping[i * active_states_limit + idx] =
              active_states[input[start - 1]][idx];
        }
        for (int idx = active_states[input[start - 1]].size();
             idx < active_states_limit; idx++) {
          mapping[i * active_states_limit + idx] = -1;
        }
      }
    }
  } else {
    active_states_limit = dfa->size;
  }

  int final = hostFE(dfa_trans, mapping, num_threads, dfa->size, input.c_str(),
                     input.size(), active_states_limit);
  return dfa->accept_state.test(final);
}
