#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "framework.hpp"
#include "regex.hpp"

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

class EnumerationCUDAMatcher final : public Matcher {
public:
  EnumerationCUDAMatcher(std::shared_ptr<const DFA> dfa) : Matcher(dfa) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    max_threads_per_block = props.maxThreadsPerBlock; // 1024
    num_sm = props.multiProcessorCount;               // 128
  }

  const std::string name() override { return "Basic CUDA"; }
  bool match(const ustring &text) override;
  void start_timer();
  double stop_timer();

private:
  int max_threads_per_block;
  int num_sm;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
      end_time;
};
