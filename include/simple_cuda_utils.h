#ifndef SIMPLE_CUDA_UTILS_H
#define SIMPLE_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

inline void cudaCheckImpl(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", file, line,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define cudaCheck(err) (cudaCheckImpl((err), __FILE__, __LINE__))

inline void randomize_matrix(float *mat, int n) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  std::srand(static_cast<unsigned int>(time.tv_usec));

  for (int i = 0; i < n; i++) {
    float tmp = static_cast<float>(std::rand() % 5) +
                0.01f * static_cast<float>(std::rand() % 5);
    tmp = (std::rand() % 2 == 0) ? tmp : -tmp;
    mat[i] = tmp;
  }
}

#endif
