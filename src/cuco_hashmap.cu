#include <iostream>
#include "common.h"
#include "cuco_hashmap.cuh"
#include "cuco_hashmap.h"

namespace pycuco {

CUCOHashmapWrapper::CUCOHashmapWrapper(torch::Tensor keys, torch::Tensor values,
                                       double load_factor) {
  CHECK_CUDA(keys);
  CHECK_CUDA(values);
  key_type_ = keys.dtype();
  value_type_ = values.dtype();

  INTEGER_TYPE_SWITCH(key_type_, Key, {
    INTEGER_TYPE_SWITCH(value_type_, Value, {
      map_ = new CUCOHashmap<Key, Value>(keys, values, load_factor);
    });
  });
}

torch::Tensor CUCOHashmapWrapper::query(torch::Tensor requests) {
  CHECK_CUDA(requests);
  INTEGER_TYPE_SWITCH(key_type_, Key, {
    INTEGER_TYPE_SWITCH(value_type_, Value, {
      auto map = (CUCOHashmap<Key, Value>*)map_;
      return map->query(requests.to(key_type_));
    });
  });

  return torch::Tensor();
}

CUCOHashmapWrapper::~CUCOHashmapWrapper() {
  INTEGER_TYPE_SWITCH(key_type_, Key, {
    INTEGER_TYPE_SWITCH(value_type_, Value,
                        { delete (CUCOHashmap<Key, Value>*)map_; });
  });
}

}  // namespace pycuco