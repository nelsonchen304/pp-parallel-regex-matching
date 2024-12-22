#include <omp.h>

#include "enumeration_omp.hpp"

void EnumerationMatcher::start_timer() {
  start_time = std::chrono::high_resolution_clock::now();
}

double EnumerationMatcher::stop_timer() {
  end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end_time - start_time)
      .count();
}

bool EnumerationMatcher::match(const ustring &text) {
  int n = text.size();
  int num_states = dfa->size;
  int block_size = (n + thread_num - 1) / thread_num;

  omp_set_num_threads(thread_num);
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int start_pos = thread_id * block_size;
    int end_pos = std::min(start_pos + block_size, n);

    for (int k = 0; k < num_states; ++k) {
      int current_state = k;

      for (int i = start_pos; i < end_pos; ++i) {
        unsigned int char_code = text[i];
        current_state = dfa->transitions[current_state][char_code];
        if (current_state == -1) {
          break;
        }
      }

      L[thread_id][k] = current_state;
    }
  }

  int state = 0;
  for (size_t i = 0; i < thread_num && state != -1; ++i) {
    state = L[i][state];
  }

  return state != -1 && dfa->accept_state.test(state);
}
