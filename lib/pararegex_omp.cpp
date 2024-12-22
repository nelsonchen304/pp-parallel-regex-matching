#include <bitset>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <utility>

#include "pararegex_omp.hpp"

using namespace std;

OmpParaMatcher::OmpParaMatcher(std::shared_ptr<const DFA> dfa,
                               size_t num_threads)
    : Matcher(dfa), num_threads(num_threads), dfa_col(ALPHABET_SIZE),
      // Initialize as -1 with size id_to_mapping[thread_num][dfa->size]
      MSU_set_local(num_threads - 1) {
  initialize_dfa_col();
  build_MSUs();
  init_MSU_set_local();
  // id_to_mapping.resize(num_threads - 1, vector<int>(dfa->size, -1));
  id_to_mapping = new int *[num_threads - 1];
  for (size_t i = 0; i < num_threads - 1; ++i) {
    id_to_mapping[i] = new int[dfa->size]();
    for (int j = 0; j < dfa->size; ++j) {
      id_to_mapping[i][j] = -1;
    }
  }
}

void OmpParaMatcher::initialize_dfa_col() {
  for (int i = 0; i < dfa->size; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      if (dfa->transitions[i][j] != -1) {
        dfa_col[j].emplace_back(i, dfa->transitions[i][j]);
      }
    }
  }
}

void OmpParaMatcher::build_MSUs() {
  for (int c = 0; c < ALPHABET_SIZE; ++c) {
    bitset<MAX_NFA_STATES> state_status;
    state_status.reset();
    for (auto &transition : dfa_col[c]) {
      if (!state_status.test(transition.second)) {
        bitset<MAX_NFA_STATES> init_mapping;
        init_mapping.set(transition.first);
        MSU_set[c].push_back({transition.second, init_mapping});
        state_status.set(transition.second);
      } else {
        for (auto &MSU : MSU_set[c]) {
          if (MSU.id == transition.second) {
            MSU.mapping.set(transition.second);
          }
        }
      }
    }
  }
}

void OmpParaMatcher::init_MSU_set_local() {
  for (size_t i = 0; i < num_threads - 1; ++i) {
    MSU_set_local[i].clear();
    for (int s = 0; s < dfa->size; ++s) {
      bitset<MAX_NFA_STATES> mapping;
      mapping.set(s);
      MSU_set_local[i].push_back({s, mapping});
    }
  }
}

bool OmpParaMatcher::match(const ustring &input) {
  int n = input.size();
  int id_final = 0;

  // reset_MSUs();

  // chrono::high_resolution_clock::time_point t1 =
  // chrono::high_resolution_clock::now(); vector<pair<int, int>>
  // decrease[num_threads - 1];

#pragma omp parallel num_threads(num_threads)
  {
    int tid_ = omp_get_thread_num();
    if (tid_ < 0) {
      std::cerr << "tid_ < 0\n";
      exit(1);
    }
    size_t tid = tid_;
    int chunk_size =
        tid < n % num_threads ? n / num_threads + 1 : n / num_threads;
    int start = tid < n % num_threads
                    ? tid * (n / num_threads + 1)
                    : tid * (n / num_threads) + n % num_threads;
    int end = min(start + chunk_size, n);

    if (tid == 0) {

      int state_cur = 0;
      for (int idx = start; idx < end; ++idx) {
        state_cur = dfa->transitions[state_cur][input[idx]];
      }
      id_final = state_cur;
    }

    else {
      MSU_set_local[tid - 1] = MSU_set[input[start - 1]];
      // decrease[tid - 1].clear();
      // decrease[tid - 1].push_back({start - start, MSU_set_local[tid -
      // 1].size()});
      for (int idx = start; idx < end; ++idx) {
        // if(MSU_set_local[tid - 1].size() != decrease[tid - 1].back().second){
        //     decrease[tid - 1].push_back({idx - start, MSU_set_local[tid -
        //     1].size()});
        // }

        for (auto &MSU : MSU_set_local[tid - 1]) {
          MSU.id = dfa->transitions[MSU.id][input[idx]];
        }
        // union those MSUs with the same id by or there mapping together
        // not using unordered_map but other hash method
        for (size_t i = 0; i < MSU_set_local[tid - 1].size(); ++i) {
          for (size_t j = i + 1; j < MSU_set_local[tid - 1].size(); ++j) {
            if (MSU_set_local[tid - 1][i].id == MSU_set_local[tid - 1][j].id) {
              MSU_set_local[tid - 1][i].mapping |=
                  MSU_set_local[tid - 1][j].mapping;
              MSU_set_local[tid - 1].erase(MSU_set_local[tid - 1].begin() + j);
              j--;
            }
          }
        }

        // for(int i = 0; i < MSU_set_local[tid - 1].size(); ++i){
        //     if(id_to_mapping[tid - 1][MSU_set_local[tid - 1][i].id] == -1){
        //         id_to_mapping[tid - 1][MSU_set_local[tid - 1][i].id] = i;
        //     }
        //     else{
        //         MSU_set_local[tid - 1][id_to_mapping[tid -
        //         1][MSU_set_local[tid - 1][i].id]].mapping |=
        //         MSU_set_local[tid - 1][i].mapping; MSU_set_local[tid -
        //         1].erase(MSU_set_local[tid - 1].begin() + i); i--;
        //     }
        // }

        // for (auto &MSU : MSU_set_local[tid - 1]) {
        //     id_to_mapping[tid - 1][MSU.id] = -1;
        // }

        // unordered_map<int, bitset<MAX_NFA_STATES>> id_to_mapping;
        // // state_status.reset();
        // for (auto &MSU : MSU_set_local[tid - 1]) {
        //     if (!state_status.test(MSU.id)) {
        //         id_to_mapping[MSU.id] = MSU.mapping;
        //         state_status.set(MSU.id);
        //     } else {
        //         id_to_mapping[MSU.id] |= MSU.mapping;
        //     }
        // }
        // MSU_set_local[tid - 1].clear();
        // for (auto &entry : id_to_mapping) {
        //     MSU_set_local[tid - 1].push_back({entry.first, entry.second});
        // }
      }
    }

    // vector<bitset<MAX_NFA_STATES>> tmp_MSUs(MAX_NFA_STATES);
    // for (int i = 0; i < MAX_NFA_STATES; ++i) {
    //     tmp_MSUs[i].reset();
    // }
    // bitset<MAX_NFA_STATES> state_status;
    // bitset<MAX_NFA_STATES> tmp_state_status;
    // state_status.set();
    // for (int i = start; i < end; ++i) {
    //     if(tid == 1){
    //         cout << "active_states[" << tid << "]: " <<
    //         active_states[tid].size() << endl;
    //     }
    //     set<int> tmp;
    //     tmp_state_status.reset();
    //     if (active_states[tid].empty() || active_states[tid].size() >
    //     dfa_col[input[i]].size()) {
    //         for (auto &transition : dfa_col[input[i]]) {
    //             if (state_status.test(transition.first)) {
    //                 tmp_MSUs[tid][transition.second] |=
    //                 MSUs[tid][transition.first];
    //                 tmp.insert(transition.second);
    //             }
    //         }
    //     } else {
    //         for (auto &state : active_states[tid]) {
    //             if (dfa->transitions[state][input[i]] != -1) {
    //                 tmp_MSUs[tid][dfa->transitions[state][input[i]]] |=
    //                 MSUs[tid][state];
    //                 tmp.insert(dfa->transitions[state][input[i]]);
    //             }
    //         }
    //     }

    //     active_states[tid] = tmp;
    //     for (auto &state : active_states[tid]) {
    //         MSUs[tid][state] = tmp_MSUs[tid][state];
    //         tmp_state_status.set(state);
    //     }
    //     state_status = tmp_state_status;
    // }
  }

  // chrono::high_resolution_clock::time_point t2 =
  // chrono::high_resolution_clock::now(); chrono::duration<double> time_span =
  // chrono::duration_cast<chrono::duration<double>>(t2 - t1); cout << "match"
  // << " takes " << time_span.count() << " seconds." << endl;

  // for (int i = 0; i < num_threads - 1; ++i) {
  //     set<int> tmp;
  //     for (auto &state : active_states[i + 1]) {
  //         bool flag = false;
  //         for (auto &state_prev : active_states[i]) {
  //             if (MSUs[i + 1][state].test(state_prev)) {
  //                 if (!flag) {
  //                     MSUs[i + 1][state] = MSUs[i][state_prev];
  //                     flag = true;
  //                 } else {
  //                     MSUs[i + 1][state] |= MSUs[i][state_prev];
  //                 }
  //                 tmp.insert(state);
  //             }
  //         }
  //     }
  //     active_states[i + 1] = tmp;
  // }
  // t1 = chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_threads - 1; ++i) {
    for (auto &MSU : MSU_set_local[i]) {
      if (MSU.mapping.test(id_final)) {
        id_final = MSU.id;
        break;
      }
    }
  }
  // t2 = chrono::high_resolution_clock::now();

  // time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "merge" << " takes " << time_span.count() << " seconds." << endl;

  // vector<vector<double>> final_decrease(dfa -> size);
  // for(int i = 0; i < num_threads - 1; ++i){
  //     int chunk_size = i < n % num_threads ? n / num_threads + 1 : n /
  //     num_threads; pair<int, int> last = decrease[i].back();
  //     final_decrease[last.second].push_back((double)last.first / chunk_size)
  //     ;
  // }
  // for(int i = 0; i < dfa -> size; ++i){
  //     if(final_decrease[i].size() != 0){
  //         cout << "state" << i << ": ";
  //         for(auto &entry : final_decrease[i]){
  //             cout << entry << ", ";
  //         }
  //         cout << endl;
  //     }
  // }

  if (dfa->accept_state.test(id_final)) {
    // cout << "Matched" << endl;
    return true;
  }
  // cout << "No match" << endl;
  return false;
}
