#pragma once
#include <torch/extension.h>

namespace cnpy {
// return dtype, shape, offset, fortran_order
std::tuple<torch::ScalarType, std::vector<int64_t>, int64_t, bool>
parse_npy_header(FILE* fp);
}