#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "advanced_cuda_kernel.hpp"

__global__ void simulate(int *dfa, unsigned char *input, int *mapping,
                         int dfa_size, int chunk_size, int threads_num, int end,
                         int a_limit

) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int state_cur = mapping[x * a_limit + y];
  if (state_cur == -1) {
    return;
  }

  int start = x * chunk_size;
  end = (x == threads_num - 1) ? end : (x + 1) * chunk_size;

  for (int i = start; i < end; i++) {
    state_cur = dfa[state_cur * 256 + input[i]];
  }
  mapping[x * a_limit + y] = state_cur;
}

// Host front-end function that allocates the memory and launches the GPU kernel
int hostFE(int *dfa, int *mapping, size_t num_threads, int dfa_size,
           const unsigned char *input, int input_size, int a_limit) {
  int n = input_size;
  int chunk_size = n / num_threads;

  dim3 threads_per_block(a_limit);
  dim3 num_blocks(num_threads);

  int *d_dfa;
  unsigned char *d_input;
  int *d_mapping;
  int h_mapping[num_threads * a_limit];
  memcpy(h_mapping, mapping, num_threads * a_limit * sizeof(int));

  cudaMalloc(&d_dfa, dfa_size * 256 * sizeof(int));
  cudaMalloc(&d_input, n * sizeof(unsigned char));
  cudaMalloc(&d_mapping, num_threads * a_limit * sizeof(int));

  cudaMemcpy(d_dfa, dfa, dfa_size * 256 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input, n * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mapping, h_mapping, num_threads * a_limit * sizeof(int),
             cudaMemcpyHostToDevice);

  simulate<<<num_blocks, threads_per_block>>>(
      d_dfa, d_input, d_mapping, dfa_size, chunk_size, num_threads, n, a_limit);
  cudaDeviceSynchronize();

  cudaMemcpy(mapping, d_mapping, num_threads * a_limit * sizeof(int),
             cudaMemcpyDeviceToHost);

  int state = 0;
  for (size_t i = 0; i < num_threads; i++) {
    int idx = 0;
    for (int j = 0; j < a_limit; j++) {
      if (h_mapping[i * a_limit + j] == state) {
        idx = j;
        break;
      }
    }
    state = mapping[i * a_limit + idx];
  }

  // std::chrono::high_resolution_clock::time_point t4 =
  // std::chrono::high_resolution_clock::now();

  // printf("Final state: %d\n", state);
  // //using double instead of float
  // std::chrono::duration<double> time_span =
  // std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  // printf("Kernel execution time: %f seconds\n", time_span.count());
  // time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t3 -
  // t2); printf("Copying back time: %f seconds\n", time_span.count());
  // time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t4 -
  // t3); printf("Final state calculation time: %f seconds\n",
  // time_span.count());
  cudaFree(d_dfa);
  cudaFree(d_input);
  cudaFree(d_mapping);

  return state;
}
