#pragma once
#include <torch/extension.h>

namespace pycuco {

class Hashmap {
 public:
  int64_t memory_usage_;
  int64_t capacity_;
};

class CUCOHashmapWrapper {
 public:
  CUCOHashmapWrapper(torch::Tensor keys, torch::Tensor values,
                     double load_factor);
  ~CUCOHashmapWrapper();
  torch::Tensor query(torch::Tensor requests);

  int64_t get_capacity() { return map_->capacity_; }
  int64_t get_memory_usage() { return map_->memory_usage_; }

 private:
  Hashmap* map_;
  caffe2::TypeMeta key_type_;
  caffe2::TypeMeta value_type_;
};

}  // namespace pycuco