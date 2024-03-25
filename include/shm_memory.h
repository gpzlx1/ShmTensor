#pragma once
#include "common.h"

namespace shm {

std::tuple<int64_t, int64_t> create_shared_mem(std::string name, int64_t size,
                                               bool pin_memory) {
  int flag = O_RDWR | O_CREAT;
  int fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK(fd != -1) << "fail to open " << name << ": " << strerror(errno);
  // Shared memory cannot be deleted if the process exits abnormally in Linux.
  int res = ftruncate(fd, (size_t)size);
  CHECK(res != -1) << "Failed to truncate the file. " << strerror(errno);
  void *ptr =
      mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(ptr != MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  if (pin_memory) {
    CUDA_CALL(cudaHostRegister(ptr, (size_t)size, cudaHostRegisterDefault));
  }
  return std::make_tuple((int64_t)ptr, (int64_t)fd);
}

std::tuple<int64_t, int64_t> open_shared_mem(std::string name, int64_t size,
                                             bool pin_memory) {
  int flag = O_RDWR;
  int fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK(fd != -1) << "fail to open " << name << ": " << strerror(errno);
  void *ptr =
      mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(ptr != MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  if (pin_memory) {
    CUDA_CALL(cudaHostRegister(ptr, (size_t)size, cudaHostRegisterDefault));
  }
  return std::make_tuple((int64_t)ptr, (int64_t)fd);
}

void release_shared_mem(std::string name, int64_t size, int64_t ptr, int64_t fd,
                        bool pin_memory) {
  if (pin_memory) {
    CUDA_CALL(cudaHostUnregister((void *)ptr));
  }
  CHECK(munmap((void *)ptr, (size_t)size) != -1) << strerror(errno);
  close((int)fd);
  shm_unlink(name.c_str());
}

torch::Tensor open_shared_tensor(int64_t ptr, pybind11::object dtype,
                                 std::vector<int64_t> shape) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(dtype);
  return torch::from_blob((void *)ptr, shape,
                          torch::TensorOptions().dtype(type));
}

}  // namespace shm