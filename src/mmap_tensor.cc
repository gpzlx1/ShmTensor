#include <string>

#include "cnpy.h"
#include "common.h"
#include "mmap_tensor.h"

namespace shm {

// return tensor, mmap_ptr, mmap_size, fd
std::tuple<torch::Tensor, int64_t, int64_t, int64_t> open_mmap_tensor(
    std::string fname, bool pin_memory) {
  FILE* fp = fopen(fname.c_str(), "rb");
  if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

  torch::ScalarType dtype;
  std::vector<int64_t> shape;
  int64_t offset;
  bool fortran_order;
  std::tie(dtype, shape, offset, fortran_order) = cnpy::parse_npy_header(fp);
  CHECK(!fortran_order) << "fortran order is not supported";
  // close fp
  fclose(fp);

  // begin mmap the file
  //// open file with open
  int fd = open(fname.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  CHECK(fd != -1) << "open file failed: " << fname;

  // begin mmap with offset
  //// get file size
  size_t file_size = lseek(fd, 0, SEEK_END);
  void* ptr = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  //// return tensor
  if (offset % 8 != 0) {
    LOG(WARNING) << "offset is not multiple of 8, may cause performance issue";
  }

  torch::Tensor tensor = torch::from_blob(ptr + offset, shape,
                                          torch::TensorOptions().dtype(dtype));

  if (pin_memory) {
    // to simpile, we just register the whole file
    CUDA_CALL(cudaHostRegister(ptr, file_size, cudaHostRegisterDefault));
  }
  return std::make_tuple(tensor, (int64_t)file_size, (int64_t)ptr, (int64_t)fd);
}

void close_mmap_tensor(int64_t size, int64_t ptr, int64_t fd, bool pin_memory) {
  if (pin_memory) {
    CUDA_CALL(cudaHostUnregister((void*)ptr));
  }
  CHECK(munmap((void*)ptr, (size_t)size) != -1) << strerror(errno);
  close((int)fd);
}

}  // namespace  shm
