#include <cuda.h>

#include "basic_cuda.hpp"
#include "regex.hpp"

void EnumerationCUDAMatcher::start_timer() {
  start_time = std::chrono::high_resolution_clock::now();
}

double EnumerationCUDAMatcher::stop_timer() {
  end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end_time - start_time)
      .count();
}

__global__ void process_states(const char *text, int n, const int *transitions,
                               int num_states, int *L, int block_size,
                               int num_threads) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= num_threads)
    return;

  int start_pos = thread_id * block_size;
  int end_pos = min(start_pos + block_size, n);

  if (start_pos >= n) {
    for (int k = 0; k < num_states; ++k) {
      L[thread_id * num_states + k] = k;
    }
    return;
  }

  for (int k = 0; k < num_states; ++k) {
    int current_state = k;
    for (int i = start_pos; i < end_pos; ++i) {
      unsigned int current_char = static_cast<unsigned char>(text[i]);
      current_state = transitions[current_state * ALPHABET_SIZE + current_char];
      if (current_state == -1) {
        break;
      }
    }
    L[thread_id * num_states + k] = current_state;
  }
}

bool EnumerationCUDAMatcher::match(const ustring &text) {
  int n = text.size();
  int num_states = dfa->size;

  max_threads_per_block = 32;
  int num_blocks = (n + max_threads_per_block - 1) / max_threads_per_block;

  int max_blocks = num_sm * 32; // Assuming 32 blocks per SM
  int actual_blocks = std::min(num_blocks, max_blocks);

  // num_sm * 32 * max_threads_per_block = 128 * 32 * 1024 = 4194304
  int actual_num_thread = actual_blocks * max_threads_per_block;
  int actual_block_size = (n + actual_num_thread - 1) / actual_num_thread;

  char *d_text;
  int *d_transitions;
  int *d_L;
  CUDA_CHECK(cudaMalloc(&d_text, n * sizeof(char)));
  CUDA_CHECK(
      cudaMalloc(&d_transitions, num_states * ALPHABET_SIZE * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_L, actual_num_thread * num_states * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_text, text.data(), n * sizeof(char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_transitions, dfa->transitions,
                        num_states * ALPHABET_SIZE * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  process_states<<<actual_blocks, max_threads_per_block>>>(
      d_text, n, d_transitions, num_states, d_L, actual_block_size,
      actual_num_thread);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  size_t L_size = actual_num_thread * num_states;

  std::vector<int> L_host(L_size);
  CUDA_CHECK(cudaMemcpy(L_host.data(), d_L, L_size * sizeof(int),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_text));
  CUDA_CHECK(cudaFree(d_transitions));
  CUDA_CHECK(cudaFree(d_L));

  int state = 0;
  int num_to_reduce = n < actual_num_thread ? n : actual_num_thread;
  for (int i = 0; i < num_to_reduce && state != -1; ++i) {
    state = L_host[i * num_states + state];
  }

  return state != -1 && dfa->accept_state.test(state);
}
