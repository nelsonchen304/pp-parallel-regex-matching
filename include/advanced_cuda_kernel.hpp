#pragma once
#include <cstddef>

int hostFE(int *dfa, int *mapping, size_t num_threads, int dfa_size,
           const unsigned char *input, int input_size, int a_limit);
