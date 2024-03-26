#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cnpy.h"
#include "mmap_tensor.h"
#include "shm_memory.h"
#include "utils.h"
using namespace shm;

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
           py::arg("ptr"), py::arg("fd"), py::arg("pin_memory"));
}