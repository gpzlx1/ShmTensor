#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <torch/extension.h>
#include <unistd.h>

#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

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

#define INTEGER_TYPE_SWITCH(val, IdType, ...)        \
  do {                                               \
    if ((val) == torch::kInt32) {                    \
      typedef int32_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else if ((val) == torch::kInt64) {             \
      typedef int64_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else {                                         \
      LOG(FATAL) << "ID can only be int32 or int64"; \
    }                                                \
  } while (0);

#define DATA_TYPE_SWITCH(val, IdType, ...)                  \
  do {                                                      \
    if ((val) == torch::kFloat32) {                         \
      typedef float IdType;                                 \
      { __VA_ARGS__ }                                       \
    } else if ((val) == torch::kFloat64) {                  \
      typedef double IdType;                                \
      { __VA_ARGS__ }                                       \
    } else if ((val) == torch::kInt32) {                    \
      typedef int32_t IdType;                               \
      { __VA_ARGS__ }                                       \
    } else if ((val) == torch::kInt64) {                    \
      typedef int64_t IdType;                               \
      { __VA_ARGS__ }                                       \
    } else if ((val) == torch::kInt8) {                     \
      typedef int8_t IdType;                                \
      { __VA_ARGS__ }                                       \
    } else if ((val) == torch::kUInt8) {                    \
      typedef uint8_t IdType;                               \
      { __VA_ARGS__ }                                       \
    } else if ((val) == torch::kInt16) {                    \
      typedef int16_t IdType;                               \
      { __VA_ARGS__ }                                       \
    } else {                                                \
      LOG(FATAL) << "value can only be float32 or float64"; \
    }                                                       \
  } while (0);
