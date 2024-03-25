#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <torch/extension.h>
#include <unistd.h>

#define CUDA_CALL(call)                                                  \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }