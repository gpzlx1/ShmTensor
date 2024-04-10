#include <vector>
#include "cache_operator.h"
#include "cuda_runtime.h"

void PinMemory(torch::Tensor& tensor) {
  CUDA_CALL(cudaHostRegister(tensor.data_ptr(), tensor.nbytes(),
                             cudaHostRegisterPortable));
}

void UnpinMemory(torch::Tensor& tensor) {
  CUDA_CALL(cudaHostUnregister(tensor.data_ptr()));
}

template <typename T, typename IndexType>
__global__ void OneDimFetchKernel(T* __restrict__ input,
                                  IndexType* __restrict__ indices,
                                  T* __restrict__ output, int64_t numel) {
  // thread id
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = gridDim.x * blockDim.x;
  for (int64_t i = thread_id; i < numel; i += threads_num) {
    output[i] = input[indices[i]];
  }
}

template <typename T, typename IndexType, int WARP_SIZE = 32>
__global__ void MultiDimFetchKernel(T* __restrict__ input,
                                    IndexType* __restrict__ indices,
                                    T* __restrict__ output, int64_t dim,
                                    int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int thread_id_in_warp = thread_id % WARP_SIZE;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;

  for (int64_t i = warp_id; i < numel; i += warp_num) {
    int64_t index = indices[i];
    int64_t ouput_start = i * dim;
    int64_t input_start = index * dim;
    for (int64_t j = thread_id_in_warp; j < dim; j += WARP_SIZE) {
      output[ouput_start + j] = input[input_start + j];
    }
  }
}

torch::Tensor UVATensorFetch(torch::Tensor& uva_data, torch::Tensor& indices) {
  int64_t numel = indices.numel();

  torch::Tensor output;

  // checkout one dim or multi dim
  if (uva_data.dim() == 1) {
    output = torch::empty(numel, uva_data.options().device(torch::kCUDA));
    DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
      INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
        OneDimFetchKernel<T, IndexType><<<(numel + 1024 - 1) / 1024, 1024>>>(
            uva_data.data_ptr<T>(), indices.data_ptr<IndexType>(),
            output.data_ptr<T>(), numel);
      });
    });

    CUDA_CALL(cudaGetLastError());
  } else {
    // compute dim and output tensor size
    int64_t dim = 1;
    std::vector<int64_t> output_dim;
    output_dim.push_back(numel);
    for (int64_t i = 1; i < uva_data.dim(); i++) {
      dim *= uva_data.size(i);
      output_dim.push_back(uva_data.size(i));
    }

    output = torch::empty(output_dim, uva_data.options().device(torch::kCUDA));
    DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
      INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
        MultiDimFetchKernel<T, IndexType><<<(numel + 1024 - 1) / 1024, 1024>>>(
            uva_data.data_ptr<T>(), indices.data_ptr<IndexType>(),
            output.data_ptr<T>(), dim, numel);
      });
    });

    CUDA_CALL(cudaGetLastError());
  }

  return output;
}