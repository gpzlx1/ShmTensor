#include <vector>
#include "cache_operator.h"
#include "cuco_hashmap.cuh"
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

template <typename Map, typename T, typename IndexType>
__global__ void OneDimCacheFetchKernel(Map map_ref, T* __restrict__ uva_data,
                                       T* __restrict__ gpu_data,
                                       IndexType* __restrict__ indices,
                                       T* __restrict__ output, int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = gridDim.x * blockDim.x;
  for (int64_t i = thread_id; i < numel; i += threads_num) {
    int64_t index = indices[i];
    auto found = map_ref.find(index);
    T* input_ptr = found != map_ref.end() ? gpu_data + found->second
                                          : uva_data + index;

    output[i] = *input_ptr;
  }
}

template <typename Map, typename T, typename IndexType, int WARP_SIZE = 32>
__global__ void MultiDimCacheFetchKernel(Map map_ref, T* __restrict__ uva_data,
                                         T* __restrict__ gpu_data,
                                         IndexType* __restrict__ indices,
                                         T* __restrict__ output, int64_t dim,
                                         int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int thread_id_in_warp = thread_id % WARP_SIZE;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;

  for (int64_t i = warp_id; i < numel; i += warp_num) {
    int64_t index = indices[i];
    T* output_ptr = output + i * dim;

    auto found = map_ref.find(index);
    T* input_ptr = found != map_ref.end() ? gpu_data + found->second * dim
                                          : uva_data + index * dim;

    for (int64_t j = thread_id_in_warp; j < dim; j += WARP_SIZE) {
      output_ptr[j] = input_ptr[j];
    }
  }
}

torch::Tensor CacheTensorFetch(torch::Tensor uva_data, torch::Tensor gpu_data,
                               torch::Tensor indices,
                               pycuco::CUCOHashmapWrapper& hashmap) {
  int64_t numel = indices.numel();
  torch::Tensor output;

  if (gpu_data.dim() == 1) {
    output = torch::empty(numel, gpu_data.options().device(torch::kCUDA));
    INTEGER_TYPE_SWITCH(hashmap.key_type_, Key, {
      INTEGER_TYPE_SWITCH(hashmap.value_type_, Value, {
        auto map = (pycuco::CUCOHashmap<Key, Value>*)hashmap.map_;
        auto map_ref = map->map_->ref(cuco::find);

        DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
          INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
            OneDimCacheFetchKernel<<<(numel + 1024 - 1) / 1024, 1024>>>(
                map_ref, uva_data.data_ptr<T>(), gpu_data.data_ptr<T>(), 
                indices.data_ptr<IndexType>(), output.data_ptr<T>(), numel);
          });
        });
      });
    });
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

    INTEGER_TYPE_SWITCH(hashmap.key_type_, Key, {
      INTEGER_TYPE_SWITCH(hashmap.value_type_, Value, {
        auto map = (pycuco::CUCOHashmap<Key, Value>*)hashmap.map_;
        auto map_ref = map->map_->ref(cuco::find);

        DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
          INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
            MultiDimCacheFetchKernel<<<(numel + 1024 - 1) / 1024, 1024>>>(
                map_ref, uva_data.data_ptr<T>(), gpu_data.data_ptr<T>(),
                indices.data_ptr<IndexType>(), output.data_ptr<T>(), dim,
                numel);
          });
        });
      });
    });

    CUDA_CALL(cudaGetLastError());
  }

  return output;
}


template <typename T, typename IndexType, typename MaskType>
__global__ void OneDimCacheFetchWithMaskKernel(
    T* __restrict__ uva_data, T* __restrict__ gpu_data,
    IndexType* __restrict__ indices, MaskType* __restrict__ mask,
    T* __restrict__ output, int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = gridDim.x * blockDim.x;
  for (int64_t i = thread_id; i < numel; i += threads_num) {
    bool found = mask[i] >= 0;
    T* input_ptr =
        found ? gpu_data + mask[i]: uva_data + indices[i];
    output[i] = *input_ptr;
  }
}

template <typename T, typename IndexType, typename MaskType, int WARP_SIZE = 32>
__global__ void MultiDimCacheFetchWithMaskKernel(
    T* __restrict__ uva_data, T* __restrict__ gpu_data,
    IndexType* __restrict__ indices, MaskType* __restrict__ mask,
    T* __restrict__ output, int64_t dim, int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int thread_id_in_warp = thread_id % WARP_SIZE;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;

  for (int64_t i = warp_id; i < numel; i += warp_num) {
    T* output_ptr = output + i * dim;

    bool found = mask[i] >= 0;
    T* input_ptr =
        found ? gpu_data + mask[i] * dim : uva_data + indices[i] * dim;

    for (int64_t j = thread_id_in_warp; j < dim; j += WARP_SIZE) {
      output_ptr[j] = input_ptr[j];
    }
  }
}

torch::Tensor CacheTensorFetchWithMask(torch::Tensor uva_data,
                                       torch::Tensor gpu_data,
                                       torch::Tensor indices,
                                       torch::Tensor mask) {
  int64_t numel = indices.numel();
  torch::Tensor output;

  if (gpu_data.dim() == 1) {
    output = torch::empty(numel, gpu_data.options().device(torch::kCUDA));
    DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
      INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
        INTEGER_TYPE_SWITCH(mask.scalar_type(), MaskType, {
          OneDimCacheFetchWithMaskKernel<<<(numel + 1024 - 1) / 1024, 1024>>>(
              uva_data.data_ptr<T>(), gpu_data.data_ptr<T>(),
              indices.data_ptr<IndexType>(), mask.data_ptr<MaskType>(),
              output.data_ptr<T>(), numel);
        });
      });
    });
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
        INTEGER_TYPE_SWITCH(mask.scalar_type(), MaskType, {
          MultiDimCacheFetchWithMaskKernel<<<(numel + 1024 - 1) / 1024, 1024>>>(
              uva_data.data_ptr<T>(), gpu_data.data_ptr<T>(),
              indices.data_ptr<IndexType>(), mask.data_ptr<MaskType>(),
              output.data_ptr<T>(), dim, numel);
        });
      });
    });

    CUDA_CALL(cudaGetLastError());
  }

  return output;
}