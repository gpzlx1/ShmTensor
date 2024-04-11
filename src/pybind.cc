#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cnpy.h"
#include "mmap_tensor.h"
#include "shm_memory.h"
#include "thrust_scan.h"
#include "utils.h"
using namespace shm;

#include <vector>
#include "cuco_hashmap.h"

#include "cache_operator.h"

PYBIND11_MODULE(ShmTensorLib, m) {
  m.def("create_shared_mem", &create_shared_mem, py::arg("name"),
        py::arg("size"), py::arg("pin_memory") = false)
      .def("open_shared_mem", &open_shared_mem, py::arg("name"),
           py::arg("size"), py::arg("pin_memory") = false)
      .def("release_shared_mem", &release_shared_mem, py::arg("name"),
           py::arg("size"), py::arg("ptr"), py::arg("fd"),
           py::arg("pin_memory") = false)
      .def("open_shared_tensor", &open_shared_tensor, py::arg("ptr"),
           py::arg("dtype"), py::arg("shape"))
      .def("file_exist", &file_exist, py::arg("filename"))
      .def("open_mmap_tensor", &open_mmap_tensor, py::arg("filename"),
           py::arg("pin_memory"))
      .def("close_mmap_tensor", &close_mmap_tensor, py::arg("size"),
           py::arg("ptr"), py::arg("fd"), py::arg("pin_memory"))
      .def("thrust_scan", &thrust_scan, py::arg("tensor"));

  m.def("pin_memory", &PinMemory, py::arg("data"))
      .def("unpin_memory", &UnpinMemory, py::arg("data"))
      .def("uva_fetch", &UVATensorFetch, py::arg("data"), py::arg("indices"))
      .def("cache_fetch", &CacheTensorFetch, py::arg("uva_data"),
           py::arg("gpu_data"), py::arg("indices"), py::arg("hashmap"))
      .def("cache_fetch_with_mask", &CacheTensorFetchWithMask,
           py::arg("uva_data"), py::arg("gpu_data"), py::arg("indices"),
           py::arg("mask"))
      .def("uva_set", &UVATensorSet, py::arg("data"), py::arg("indices"),
           py::arg("update"))
      .def("cache_set", &CacheTensorSet, py::arg("uva_data"),
           py::arg("gpu_data"), py::arg("indices"), py::arg("update"),
           py::arg("hashmap"))
      .def("cache_set_with_mask", &CacheTensorSetWithMask, py::arg("uva_data"),
           py::arg("gpu_data"), py::arg("indices"), py::arg("update"),
           py::arg("mask"))
      .def("csr_sampling", &CSRWiseSampling, py::arg("indptr"),
           py::arg("indices"), py::arg("seeds"), py::arg("num_picks"),
           py::arg("replace"))
      .def("csr_cache_sampling", &CSRWiseCacheSampling, py::arg("uva_indptr"),
           py::arg("uva_indices"), py::arg("gpu_indptr"),
           py::arg("gpu_indices"), py::arg("hashmap"), py::arg("seeds"),
           py::arg("num_picks"), py::arg("replace"))
      .def("csr_mask_sampling", &CSRWiseMaskSampling, py::arg("uva_indptr"),
           py::arg("uva_indices"), py::arg("gpu_indptr"),
           py::arg("gpu_indices"), py::arg("seeds"), py::arg("mask"),
           py::arg("num_picks"), py::arg("replace"))
      .def("create_subcsr", &CreateCacheCSR, py::arg("uva_indptr"),
           py::arg("uva_indices"), py::arg("seeds"))
      .def("tensor_relabel", &TensorRelabel, py::arg("tensors"));

  py::class_<pycuco::CUCOHashmapWrapper>(m, "CUCOStaticHashmap")
      .def(py::init<torch::Tensor, torch::Tensor, double>())
      .def("query", &pycuco::CUCOHashmapWrapper::query)
      .def("capacity", &pycuco::CUCOHashmapWrapper::get_capacity)
      .def("memory_usage", &pycuco::CUCOHashmapWrapper::get_memory_usage);
}