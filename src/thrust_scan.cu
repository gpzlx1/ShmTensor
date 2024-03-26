#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <torch/extension.h>

#include "thrust_scan.h"

void thrust_scan(torch::Tensor a) {
  int64_t numel = a.numel();
  thrust::for_each(thrust::device, thrust::counting_iterator<int64_t>(0),
                   thrust::counting_iterator<int64_t>(numel),
                   [ptr = a.data_ptr<int32_t>()] __device__(int64_t i) {
                     if (i > 0) {
                       ptr[i] += ptr[i - 1];
                     }
                   });
}
