#include <c10/cuda/CUDACachingAllocator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector>

#include "cache_operator.h"
#include "cuda_runtime.h"

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

inline __device__ int64_t AtomicCAS(int64_t *const address,
                                    const int64_t compare, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ int32_t AtomicCAS(int32_t *const address,
                                    const int32_t compare, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ int64_t AtomicMin(int64_t *const address, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMin(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int32_t AtomicMin(int32_t *const address, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMin(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

template <typename T>
struct RelabelHashmap {
  __device__ inline RelabelHashmap(T *__restrict__ Kptr, T *__restrict__ Vptr,
                                   size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Update(T key, T value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    T prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key && prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    AtomicMin(vptr + pos, value);
  }

  __device__ inline T SearchForPos(T key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) return pos;
      if (kptr[pos] == kEmptyKey) return -1;
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline T SearchForValue(T key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    while (true) {
      if (kptr[pos] == key) return vptr[pos];
      if (kptr[pos] == kEmptyKey) return -1;
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline uint32_t hash(int32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(int64_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint64_t key) { return key & (capacity - 1); }

  T kEmptyKey{-1};
  T *kptr;
  T *vptr;
  uint32_t capacity{0};
};

std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabel(
    std::vector<torch::Tensor> tensors) {
  std::vector<int64_t> split_sizes;
  for (auto d : tensors) {
    split_sizes.push_back(d.numel());
  }
  torch::Tensor total_tensor = torch::cat(tensors, 0);
  torch::Tensor relabel_tensor;
  torch::Tensor unique_tensor;

  INTEGER_TYPE_SWITCH(total_tensor.scalar_type(), T, {
    int numel = total_tensor.numel();

    // create hashmap
    int hashmap_size = 1 << static_cast<uint32_t>(std::log2(numel) + 1);
    T MAX = std::numeric_limits<T>::max();
    torch::Tensor key_tensor = torch::full(
        {hashmap_size}, -1, total_tensor.options().device(torch::kCUDA));
    torch::Tensor index_tensor = torch::full(
        {hashmap_size}, MAX, total_tensor.options().device(torch::kCUDA));

    using it = thrust::counting_iterator<T>;
    thrust::for_each(
        it(0), it(numel),
        [key = key_tensor.data_ptr<T>(), index = index_tensor.data_ptr<T>(),
         in = total_tensor.data_ptr<T>(), numel,
         hashmap_size] __device__(T i) mutable {
          RelabelHashmap<T> table(key, index, hashmap_size);
          table.Update(in[i], i);
        });

    // prefix sum
    torch::Tensor item_prefix_tensor =
        torch::empty(numel + 1, total_tensor.options());
    thrust::device_ptr<T> item_prefix(
        static_cast<T *>(item_prefix_tensor.data_ptr<T>()));
    thrust::for_each(
        it(0), it(numel),
        [key = key_tensor.data_ptr<T>(), index = index_tensor.data_ptr<T>(),
         in = total_tensor.data_ptr<T>(),
         count = thrust::raw_pointer_cast(item_prefix), numel,
         hashmap_size] __device__(T i) mutable {
          RelabelHashmap<T> table(key, index, hashmap_size);
          count[i] = table.SearchForValue(in[i]) == i ? 1 : 0;
        });
    cub_exclusiveSum<T>(thrust::raw_pointer_cast(item_prefix), numel + 1);

    // unique
    int tot = item_prefix[numel];
    unique_tensor =
        torch::empty({tot}, total_tensor.options().device(torch::kCUDA));
    torch::Tensor value_tensor = torch::empty(
        {hashmap_size}, total_tensor.options().device(torch::kCUDA));
    thrust::for_each(
        it(0), it(numel),
        [key = key_tensor.data_ptr<T>(), index = index_tensor.data_ptr<T>(),
         in = total_tensor.data_ptr<T>(),
         prefix = thrust::raw_pointer_cast(item_prefix),
         unique = unique_tensor.data_ptr<T>(),
         value = value_tensor.data_ptr<T>(), numel,
         hashmap_size] __device__(T i) mutable {
          RelabelHashmap<T> table(key, index, hashmap_size);
          T pos = table.SearchForPos(in[i]);
          if (index[pos] == i) {
            unique[prefix[i]] = in[i];
            value[pos] = prefix[i];
          }
        });

    // relabel
    relabel_tensor = torch::empty_like(total_tensor);
    thrust::for_each(
        it(0), it(numel),
        [key = key_tensor.data_ptr<T>(), value = value_tensor.data_ptr<T>(),
         in = total_tensor.data_ptr<T>(), out = relabel_tensor.data_ptr<T>(),
         hashmap_size] __device__(T i) mutable {
          RelabelHashmap<T> table(key, value, hashmap_size);
          out[i] = table.SearchForValue(in[i]);
        });
  });

  std::vector<torch::Tensor> relabel_tensors =
      relabel_tensor.split_with_sizes(split_sizes, 0);
  return std::make_tuple(unique_tensor, relabel_tensors);
}
