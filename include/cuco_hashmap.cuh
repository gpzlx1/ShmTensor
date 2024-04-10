#pragma once
#include <c10/cuda/CUDACachingAllocator.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuco/static_map.cuh>

#include "cuco_hashmap.h"

namespace pycuco {

int64_t temp_size = 0;

template <typename T>
class torch_allocator {
 public:
  using value_type = T;

  torch_allocator() = default;

  template <class U>
  torch_allocator(torch_allocator<U> const&) noexcept {}

  value_type* allocate(std::size_t n) {
    value_type* p = reinterpret_cast<value_type*>(
        torch_cuda_allocator->raw_allocate(sizeof(value_type) * n));
    temp_size += n * sizeof(value_type);
    return p;
  }

  void deallocate(value_type* p, std::size_t) {
    torch_cuda_allocator->raw_deallocate(p);
  }

 private:
  c10::cuda::CUDACachingAllocator::CUDAAllocator* torch_cuda_allocator =
      c10::cuda::CUDACachingAllocator::get();
};

template <typename T, typename U>
bool operator==(torch_allocator<T> const&, torch_allocator<U> const&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(torch_allocator<T> const& lhs,
                torch_allocator<U> const& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename Key, typename Value>
class CUCOHashmap : public Hashmap {
 public:
  using map_type = cuco::static_map<
      Key, Value, std::size_t, cuda::thread_scope_device, thrust::equal_to<Key>,
      cuco::linear_probing<4, cuco::default_hash_function<Key>>,
      torch_allocator<cuco::pair<Key, Value>>, cuco::storage<1>>;

  CUCOHashmap(torch::Tensor keys, torch::Tensor values, double load_factor) {
    Key constexpr empty_key_sentinel = -1;
    Value constexpr empty_value_sentinel = -1;

    int64_t numel = keys.numel();
    std::size_t const capacity = std::ceil(numel / load_factor);

    // Create a cuco::static_map
    temp_size = 0;
    map_ = new map_type(capacity, cuco::empty_key{empty_key_sentinel},
                        cuco::empty_value{empty_value_sentinel});
    auto zipped = thrust::make_zip_iterator(
        thrust::make_tuple(keys.data_ptr<Key>(), values.data_ptr<Value>()));
    map_->insert(zipped, zipped + numel);

    // Set property
    key_options_ = keys.options();
    value_options_ = values.options();
    capacity_ = capacity;
    memory_usage_ = temp_size;  // for test
  };

  ~CUCOHashmap() { delete map_; };

  torch::Tensor query(torch::Tensor requests) {
    int64_t numel = requests.numel();
    torch::Tensor result = torch::full_like(requests, -1, value_options_);
    map_->find(requests.data_ptr<Key>(), requests.data_ptr<Key>() + numel,
               result.data_ptr<Value>());
    return result;
  };

 private:
  torch::TensorOptions key_options_;
  torch::TensorOptions value_options_;
  map_type* map_;
};

}  // namespace pycuco