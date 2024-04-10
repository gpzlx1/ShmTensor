#include <vector>
#include "cache_operator.h"
#include "cuco_hashmap.cuh"
#include "cuda_runtime.h"

template <typename T, typename IndexType>
__global__ void OneDimSetKernel(T* __restrict__ data,
                                IndexType* __restrict__ indices,
                                T* __restrict__ update_data, int64_t numel) {
  // thread id
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = gridDim.x * blockDim.x;
  for (int64_t i = thread_id; i < numel; i += threads_num) {
    data[indices[i]] = update_data[i];
  }
}

template <typename T, typename IndexType, int WARP_SIZE = 32>
__global__ void MultiDimSetKernel(T* __restrict__ data,
                                  IndexType* __restrict__ indices,
                                  T* __restrict__ update_data, int64_t dim,
                                  int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int thread_id_in_warp = thread_id % WARP_SIZE;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;

  for (int64_t i = warp_id; i < numel; i += warp_num) {
    int64_t index = indices[i];

    T* data_ptr = data + index * dim;
    T* update_ptr = update_data + i * dim;
    for (int64_t j = thread_id_in_warp; j < dim; j += WARP_SIZE) {
      data_ptr[j] = update_ptr[j];
    }
  }
}

void UVATensorSet(torch::Tensor& uva_data, torch::Tensor& indices,
                  torch::Tensor& data) {
  int64_t numel = indices.numel();

  // checkout one dim or multi dim
  if (uva_data.dim() == 1) {
    DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
      INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
        OneDimSetKernel<T, IndexType><<<(numel + 1024 - 1) / 1024, 1024>>>(
            uva_data.data_ptr<T>(), indices.data_ptr<IndexType>(),
            data.data_ptr<T>(), numel);
      });
    });

    CUDA_CALL(cudaGetLastError());

  } else {
    // compute dim
    int64_t dim = 1;
    std::vector<int64_t> output_dim;
    output_dim.push_back(numel);
    for (int64_t i = 1; i < uva_data.dim(); i++) {
      dim *= uva_data.size(i);
      output_dim.push_back(uva_data.size(i));
    }

    DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
      INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
        MultiDimSetKernel<T, IndexType><<<(numel + 1024 - 1) / 1024, 1024>>>(
            uva_data.data_ptr<T>(), indices.data_ptr<IndexType>(),
            data.data_ptr<T>(), dim, numel);
      });
    });

    CUDA_CALL(cudaGetLastError());
  }
}

template <typename Map, typename T, typename IndexType, int WARP_SIZE = 32>
__global__ void MultiDimCacheSetKernel(Map map_ref, T* __restrict__ uva_data,
                                       T* __restrict__ gpu_data,
                                       IndexType* __restrict__ indices,
                                       T* __restrict__ update_data, int64_t dim,
                                       int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int thread_id_in_warp = thread_id % WARP_SIZE;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;

  for (int64_t i = warp_id; i < numel; i += warp_num) {
    int64_t index = indices[i];
    T* update_ptr = update_data + i * dim;

    auto found = map_ref.find(index);
    T* data_ptr = found != map_ref.end() ? gpu_data + found->second * dim
                                         : uva_data + index * dim;

    for (int64_t j = thread_id_in_warp; j < dim; j += WARP_SIZE) {
      data_ptr[j] = update_ptr[j];
    }
  }
}

void CacheTensorSet(torch::Tensor uva_data, torch::Tensor gpu_data,
                    torch::Tensor indices, torch::Tensor data,
                    pycuco::CUCOHashmapWrapper& hashmap) {
  int64_t numel = indices.numel();

  if (gpu_data.dim() == 1) {
    throw std::runtime_error("Not implemented for dim = 1");

  } else {
    // compute dim
    int64_t dim = 1;
    std::vector<int64_t> output_dim;
    output_dim.push_back(numel);
    for (int64_t i = 1; i < uva_data.dim(); i++) {
      dim *= uva_data.size(i);
      output_dim.push_back(uva_data.size(i));
    }

    INTEGER_TYPE_SWITCH(hashmap.key_type_, Key, {
      INTEGER_TYPE_SWITCH(hashmap.value_type_, Value, {
        auto map = (pycuco::CUCOHashmap<Key, Value>*)hashmap.map_;
        auto map_ref = map->map_->ref(cuco::find);

        DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
          INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
            MultiDimCacheSetKernel<<<(numel + 1024 - 1) / 1024, 1024>>>(
                map_ref, uva_data.data_ptr<T>(), gpu_data.data_ptr<T>(),
                indices.data_ptr<IndexType>(), data.data_ptr<T>(), dim, numel);
          });
        });
      });
    });

    CUDA_CALL(cudaGetLastError());
  }
}

template <typename T, typename IndexType, typename MaskType, int WARP_SIZE = 32>
__global__ void MultiDimCacheSetWithMaskKernel(T* __restrict__ uva_data,
                                               T* __restrict__ gpu_data,
                                               IndexType* __restrict__ indices,
                                               MaskType* __restrict__ mask,
                                               T* __restrict__ update_data,
                                               int64_t dim, int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int thread_id_in_warp = thread_id % WARP_SIZE;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;

  for (int64_t i = warp_id; i < numel; i += warp_num) {
    T* update_ptr = update_data + i * dim;

    bool found = mask[i] >= 0;
    T* data_ptr =
        found ? gpu_data + mask[i] * dim : uva_data + indices[i] * dim;

    for (int64_t j = thread_id_in_warp; j < dim; j += WARP_SIZE) {
      data_ptr[j] = update_ptr[j];
    }
  }
}

void CacheTensorSetWithMask(torch::Tensor uva_data, torch::Tensor gpu_data,
                            torch::Tensor indices, torch::Tensor data,
                            torch::Tensor mask) {
  int64_t numel = indices.numel();

  if (gpu_data.dim() == 1) {
    throw std::runtime_error("Not implemented for dim = 1");

  } else {
    // compute dim
    int64_t dim = 1;
    std::vector<int64_t> output_dim;
    output_dim.push_back(numel);
    for (int64_t i = 1; i < uva_data.dim(); i++) {
      dim *= uva_data.size(i);
      output_dim.push_back(uva_data.size(i));
    }

    DATA_TYPE_SWITCH(uva_data.scalar_type(), T, {
      INTEGER_TYPE_SWITCH(indices.scalar_type(), IndexType, {
        INTEGER_TYPE_SWITCH(mask.scalar_type(), MaskType, {
          MultiDimCacheSetWithMaskKernel<<<(numel + 1024 - 1) / 1024, 1024>>>(
              uva_data.data_ptr<T>(), gpu_data.data_ptr<T>(),
              indices.data_ptr<IndexType>(), mask.data_ptr<MaskType>(),
              data.data_ptr<T>(), dim, numel);
        });
      });
    });

    CUDA_CALL(cudaGetLastError());
  }
}