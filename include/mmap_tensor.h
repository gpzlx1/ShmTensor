#pragma once
#include <torch/extension.h>

namespace shm {

// return tensor, mmap_ptr, mmap_size, fd
std::tuple<torch::Tensor, int64_t, int64_t, int64_t> open_mmap_tensor(
    std::string fname, bool pin_memory);

void close_mmap_tensor(int64_t size, int64_t ptr, int64_t fd, bool pin_memory);

}  // namespace  shm
