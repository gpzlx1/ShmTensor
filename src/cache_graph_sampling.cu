#include <c10/cuda/CUDACachingAllocator.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector>

#include "cache_operator.h"
#include "cuco_hashmap.cuh"
#include "cuda_runtime.h"

inline __device__ int64_t AtomicMax(int64_t *const address, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int32_t AtomicMax(int32_t *const address, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

template <typename T>
inline void cub_exclusiveSum(T *arrays, int64_t array_length) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);

  c10::cuda::CUDACachingAllocator::CUDAAllocator *cuda_allocator =
      c10::cuda::CUDACachingAllocator::get();
  d_temp_storage = cuda_allocator->raw_allocate(temp_storage_bytes);

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);

  cuda_allocator->raw_deallocate(d_temp_storage);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetSubIndptr(
    torch::Tensor indptr, torch::Tensor seeds, int64_t num_pick, bool replace) {
  int64_t numel = seeds.numel();
  torch::Tensor sub_indptr =
      torch::empty((numel + 1), indptr.options().device(torch::kCUDA));
  torch::Tensor indices_begin =
      torch::empty(numel, indptr.options().device(torch::kCUDA));
  torch::Tensor indices_end =
      torch::empty(numel, indptr.options().device(torch::kCUDA));

  INTEGER_TYPE_SWITCH(indptr.scalar_type(), IdType, {
    INTEGER_TYPE_SWITCH(seeds.scalar_type(), IndexType, {
      thrust::device_ptr<IdType> item_prefix(
          static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));

      using it = thrust::counting_iterator<IdType>;
      thrust::for_each(thrust::device, it(0), it(numel),
                       [in_indptr = indptr.data_ptr<IdType>(),
                        index = seeds.data_ptr<IndexType>(),
                        out = thrust::raw_pointer_cast(item_prefix),
                        begin_ptr = indices_begin.data_ptr<IdType>(),
                        end_ptr = indices_end.data_ptr<IdType>(), replace,
                        num_pick] __device__(int i) mutable {
                         IdType row = index[i];
                         IdType begin = in_indptr[row];
                         IdType end = in_indptr[row + 1];
                         IdType deg = end - begin;
                         begin_ptr[i] = begin;
                         end_ptr[i] = end;
                         if (replace) {
                           out[i] = deg == 0 ? 0 : num_pick;
                         } else {
                           out[i] = deg < num_pick ? deg : num_pick;
                         }
                       });

      cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix),
                               numel + 1);
    });
  });

  return std::make_tuple(sub_indptr, indices_begin, indices_end);
}

template <typename T, int WARP_SIZE = 32>
__global__ void CSRWiseSampleUniformKernel(T *__restrict__ indices_begin,
                                           T *__restrict__ indices_end,
                                           T *__restrict__ indices,
                                           T *__restrict__ sampled_indptr,
                                           T *__restrict__ sampled_indices,
                                           int64_t num_pick, int64_t numel,
                                           uint64_t rand_seed) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = thread_id / WARP_SIZE;
  int lane = thread_id % WARP_SIZE;

  curandStatePhilox4_32_10_t rng;
  // curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
  curand_init(rand_seed + lane, thread_id, 0, &rng);

  for (int i = warp_id; i < numel; i += warp_num) {
    T begin = indices_begin[i];
    T end = indices_end[i];
    T deg = end - begin;
    T *output_ptr = sampled_indices + sampled_indptr[i];
    T *input_ptr = indices + begin;

    if (deg <= num_pick) {
      // just copy
      for (int j = lane; j < deg; j += WARP_SIZE) {
        output_ptr[j] = input_ptr[j];
      }

    } else {
      // generate permutation list via reservoir algorithm
      for (int i = lane; i < num_pick; i += WARP_SIZE) {
        output_ptr[i] = i;
      }
      __syncwarp();

      for (int i = lane + num_pick; i < deg; i += WARP_SIZE) {
        int num = curand(&rng) % (i + 1);
        if (num < num_pick) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(output_ptr + num, (T)i);
        }
      }
      __syncwarp();

      // copy permutation over
      for (int i = lane; i < num_pick; i += WARP_SIZE) {
        auto perm_idx = output_ptr[i];
        output_ptr[i] = input_ptr[output_ptr[i]];
      }
    }
  }
}

template <typename T, int WARP_SIZE = 32>
__global__ void CSRWiseSampleUniformReplaceKernel(
    T *__restrict__ indices_begin, T *__restrict__ indices_end,
    T *__restrict__ indices, T *__restrict__ sampled_indptr,
    T *__restrict__ sampled_indices, int64_t num_pick, int64_t numel,
    uint64_t rand_seed) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = thread_id / WARP_SIZE;
  int lane = thread_id % WARP_SIZE;

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed + lane, thread_id, 0, &rng);

  for (int i = warp_id; i < numel; i += warp_num) {
    T begin = indices_begin[i];
    T end = indices_end[i];
    T deg = end - begin;

    if (deg > 0) {
      T *output_ptr = sampled_indices + sampled_indptr[i];
      T *input_ptr = indices + begin;
      for (int i = lane; i < num_pick; i += WARP_SIZE) {
        int select = curand(&rng) % deg;
        output_ptr[i] = input_ptr[select];
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> CSRWiseSampling(torch::Tensor indptr,
                                                         torch::Tensor indices,
                                                         torch::Tensor seeds,
                                                         int64_t num_picks,
                                                         bool replace) {
  int64_t numel = seeds.numel();
  torch::Tensor sampled_indptr;
  torch::Tensor indices_begin;
  torch::Tensor indices_end;
  std::tie(sampled_indptr, indices_begin, indices_end) =
      GetSubIndptr(indptr, seeds, num_picks, replace);
  torch::Tensor sampled_indices;

  INTEGER_TYPE_SWITCH(indptr.scalar_type(), T, {
    thrust::device_ptr<T> item_prefix(
        static_cast<T *>(sampled_indptr.data_ptr<T>()));
    int nnz = item_prefix[numel];

    // allocate sampled_indices
    sampled_indices = torch::empty(nnz, indices.options().device(torch::kCUDA));

    // set rand seeds
    // uint64_t random_seed =
    //    std::chrono::system_clock::now().time_since_epoch().count();
    // for debug
    uint64_t random_seed = 7777;

    if (replace) {
      CSRWiseSampleUniformReplaceKernel<<<(numel + 255) / 256, 256>>>(
          indices_begin.data_ptr<T>(), indices_end.data_ptr<T>(),
          indices.data_ptr<T>(), sampled_indptr.data_ptr<T>(),
          sampled_indices.data_ptr<T>(), num_picks, numel, random_seed);
    } else {
      CSRWiseSampleUniformKernel<<<(numel + 255) / 256, 256>>>(
          indices_begin.data_ptr<T>(), indices_end.data_ptr<T>(),
          indices.data_ptr<T>(), sampled_indptr.data_ptr<T>(),
          sampled_indices.data_ptr<T>(), num_picks, numel, random_seed);
    }
  });

  return std::make_tuple(sampled_indptr, sampled_indices);
}

template <typename Map, typename T, typename IndexMask, int WARP_SIZE = 32>
__global__ void CSRWiseCacheSampleUniformKernel(
    Map map_ref, T *__restrict__ indices_begin, T *__restrict__ indices_end,
    T *__restrict__ indices, T *__restrict__ gpu_indptr,
    T *__restrict__ gpu_indices, IndexMask *__restrict__ seeds,
    T *__restrict__ sampled_indptr, T *__restrict__ sampled_indices,
    int64_t num_pick, int64_t numel, uint64_t rand_seed) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = thread_id / WARP_SIZE;
  int lane = thread_id % WARP_SIZE;

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed + lane, thread_id, 0, &rng);

  for (int i = warp_id; i < numel; i += warp_num) {
    auto value = map_ref.find(seeds[i]);
    bool in_gpu = value != map_ref.end();

    T begin = in_gpu ? gpu_indptr[value->second] : indices_begin[i];
    T end = in_gpu ? gpu_indptr[value->second + 1] : indices_end[i];
    T deg = end - begin;
    T *output_ptr = sampled_indices + sampled_indptr[i];
    T *input_ptr = in_gpu ? gpu_indices + begin : indices + begin;

    if (deg <= num_pick) {
      // just copy
      for (int j = lane; j < deg; j += WARP_SIZE) {
        output_ptr[j] = input_ptr[j];
      }

    } else {
      // generate permutation list via reservoir algorithm
      for (int i = lane; i < num_pick; i += WARP_SIZE) {
        output_ptr[i] = i;
      }
      __syncwarp();

      for (int i = lane + num_pick; i < deg; i += WARP_SIZE) {
        int num = curand(&rng) % (i + 1);
        if (num < num_pick) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(output_ptr + num, (T)i);
        }
      }
      __syncwarp();

      // copy permutation over
      for (int i = lane; i < num_pick; i += WARP_SIZE) {
        auto perm_idx = output_ptr[i];
        output_ptr[i] = input_ptr[output_ptr[i]];
      }
    }
  }
}

template <typename Map, typename T, typename IndexMask, int WARP_SIZE = 32>
__global__ void CSRWiseCacheSampleUniformReplaceKernel(
    Map map_ref, T *__restrict__ indices_begin, T *__restrict__ indices_end,
    T *__restrict__ indices, T *__restrict__ gpu_indptr,
    T *__restrict__ gpu_indices, IndexMask *__restrict__ seeds,
    T *__restrict__ sampled_indptr, T *__restrict__ sampled_indices,
    int64_t num_pick, int64_t numel, uint64_t rand_seed) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = thread_id / WARP_SIZE;
  int lane = thread_id % WARP_SIZE;

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed + lane, thread_id, 0, &rng);

  for (int i = warp_id; i < numel; i += warp_num) {
    auto found = map_ref.find(seeds[i]);
    bool in_gpu = found != map_ref.end();
    T begin = in_gpu ? gpu_indptr[found->second] : indices_begin[i];
    T end = in_gpu ? gpu_indptr[found->second + 1] : indices_end[i];
    T deg = end - begin;

    if (deg > 0) {
      T *output_ptr = sampled_indices + sampled_indptr[i];
      T *input_ptr = in_gpu ? gpu_indices + begin : indices + begin;

      for (int i = lane; i < num_pick; i += WARP_SIZE) {
        int select = curand(&rng) % deg;
        output_ptr[i] = input_ptr[select];
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> CSRWiseCacheSampling(
    torch::Tensor uva_indptr, torch::Tensor uva_indices,
    torch::Tensor gpu_indptr, torch::Tensor gpu_indices,
    pycuco::CUCOHashmapWrapper &hashmap, torch::Tensor seeds, int64_t num_picks,
    bool replace) {
  int64_t numel = seeds.numel();
  torch::Tensor sampled_indptr;
  torch::Tensor indices_begin;
  torch::Tensor indices_end;
  torch::Tensor sampled_indices;

  std::tie(sampled_indptr, indices_begin, indices_end) =
      GetSubIndptr(uva_indptr, seeds, num_picks, replace);

  INTEGER_TYPE_SWITCH(hashmap.key_type_, Key, {
    INTEGER_TYPE_SWITCH(hashmap.value_type_, Value, {
      auto map = (pycuco::CUCOHashmap<Key, Value> *)hashmap.map_;
      auto map_ref = map->map_->ref(cuco::find);

      INTEGER_TYPE_SWITCH(uva_indptr.scalar_type(), T, {
        INTEGER_TYPE_SWITCH(seeds.scalar_type(), IndexType, {
          // allocate sampled_indices
          thrust::device_ptr<T> item_prefix(
              static_cast<T *>(sampled_indptr.data_ptr<T>()));
          int nnz = item_prefix[numel];
          sampled_indices =
              torch::empty(nnz, gpu_indices.options().device(torch::kCUDA));

          uint64_t random_seed = 7777;

          if (replace) {
            CSRWiseCacheSampleUniformReplaceKernel<<<(numel + 255) / 256,
                                                     256>>>(
                map_ref, indices_begin.data_ptr<T>(), indices_end.data_ptr<T>(),
                uva_indices.data_ptr<T>(), gpu_indptr.data_ptr<T>(),
                gpu_indices.data_ptr<T>(), seeds.data_ptr<IndexType>(),
                sampled_indptr.data_ptr<T>(), sampled_indices.data_ptr<T>(),
                num_picks, numel, random_seed);
          } else {
            CSRWiseCacheSampleUniformKernel<<<(numel + 255) / 256, 256>>>(
                map_ref, indices_begin.data_ptr<T>(), indices_end.data_ptr<T>(),
                uva_indices.data_ptr<T>(), gpu_indptr.data_ptr<T>(),
                gpu_indices.data_ptr<T>(), seeds.data_ptr<IndexType>(),
                sampled_indptr.data_ptr<T>(), sampled_indices.data_ptr<T>(),
                num_picks, numel, random_seed);
          }
        });
      });
    });
  });

  return std::make_tuple(sampled_indptr, sampled_indices);
}

template <typename T, typename IndexType, int WARP_SIZE = 32>
__global__ void CSRIndicesCopyKernel(T *__restrict__ uva_indptr,
                                     T *__restrict__ uva_indices,
                                     IndexType *__restrict__ seeds,
                                     T *__restrict__ gpu_indptr,
                                     T *__restrict__ gpu_indices,
                                     int64_t numel) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  int warp_id = thread_id / WARP_SIZE;
  int lane = thread_id % WARP_SIZE;

  for (int i = warp_id; i < numel; i += warp_num) {
    T begin = uva_indptr[seeds[i]];
    T end = uva_indptr[seeds[i] + 1];
    T deg = end - begin;

    if (deg > 0) {
      T *output_ptr = gpu_indices + gpu_indptr[i];
      T *input_ptr = uva_indices + begin;
      for (int i = lane; i < deg; i++) {
        output_ptr[i] = input_ptr[i];
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> CreateCacheCSR(
    torch::Tensor uva_indptr, torch::Tensor uva_indices, torch::Tensor seeds) {
  int64_t numel = seeds.numel();
  torch::Tensor gpu_indptr;
  torch::Tensor gpu_indices;

  INTEGER_TYPE_SWITCH(uva_indptr.scalar_type(), T, {
    INTEGER_TYPE_SWITCH(seeds.scalar_type(), IndexType, {
      // compute gpu_indptr
      gpu_indptr =
          torch::empty(numel + 1, uva_indptr.options().device(torch::kCUDA));

      using it = thrust::counting_iterator<IndexType>;
      thrust::for_each(
          thrust::device, it(0), it(numel),
          [in = uva_indptr.data_ptr<T>(), out = gpu_indptr.data_ptr<T>(),
           index = seeds.data_ptr<IndexType>()] __device__(T i) mutable {
            out[i] = in[index[i] + 1] - in[index[i]];
          });

      thrust::device_ptr<T> item_prefix(
          static_cast<T *>(gpu_indptr.data_ptr<T>()));
      cub_exclusiveSum<T>(thrust::raw_pointer_cast(item_prefix), numel + 1);

      // compute gpu_indices
      T nnz = item_prefix[numel];
      gpu_indices =
          torch::empty(nnz, uva_indices.options().device(torch::kCUDA));

      CSRIndicesCopyKernel<<<(numel + 255) / 256, 256>>>(
          uva_indptr.data_ptr<T>(), uva_indices.data_ptr<T>(),
          seeds.data_ptr<IndexType>(), gpu_indptr.data_ptr<T>(),
          gpu_indices.data_ptr<T>(), numel);
    });
  });

  return std::make_tuple(gpu_indptr, gpu_indices);
}