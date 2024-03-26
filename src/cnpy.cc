#include <cassert>
#include <iostream>
#include <regex>

#include "cnpy.h"

namespace cnpy {

std::unordered_map<std::string, torch::ScalarType> npy2torch = {
    {"<f2", torch::kFloat16}, {"<f4", torch::kFloat32},
    {"<f8", torch::kFloat64}, {"|u1", torch::kUInt8},
    {"|i1", torch::kInt8},    {"<i2", torch::kInt16},
    {"<i4", torch::kInt32},   {"<i8", torch::kInt64},
};

// copy from: https://github.com/rogersce/cnpy
// return dtype, shape, offset, fortran_order
std::tuple<torch::ScalarType, std::vector<int64_t>, int64_t, bool>
parse_npy_header(FILE* fp) {
  // for return
  torch::ScalarType dtype;
  std::vector<int64_t> shape;
  int64_t offset;
  bool fortran_order;

  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
  std::string header = fgets(buffer, 256, fp);
  assert(header[header.size() - 1] == '\n');

  size_t loc1, loc2;

  // fortran order
  loc1 = header.find("fortran_order");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

  // shape
  loc1 = header.find("(");
  loc2 = header.find(")");
  if (loc1 == std::string::npos || loc2 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: '(' or ')'");

  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  shape.clear();

  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  while (std::regex_search(str_shape, sm, num_regex)) {
    shape.push_back(std::stoi(sm[0].str()));
    str_shape = sm.suffix().str();
  }

  // endian, word size, data type
  // byte order code | stands for not applicable.
  // not sure when this applies except for byte array
  loc1 = header.find("descr");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'descr'");
  loc1 += 9;
  bool littleEndian =
      (header[loc1] == '<' || header[loc1] == '|' ? true : false);
  assert(littleEndian);

  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  std::string npy_type = header.substr(loc1, loc2 + 2);
  auto iter = npy2torch.find(npy_type);
  if (iter == npy2torch.end()) {
    std::cerr << "dtype: " << npy_type << std::endl;
    throw std::runtime_error("parse_npy_header: failed to find dtype");
  }
  dtype = iter->second;

  // offset
  offset = ftell(fp);

  // return
  return std::make_tuple(dtype, shape, offset, fortran_order);
}

torch::Tensor test_open_from_numpy(std::string fname) {
  FILE* fp = fopen(fname.c_str(), "rb");
  if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

  torch::ScalarType dtype;
  std::vector<int64_t> shape;
  int64_t offset;
  bool fortran_order;
  std::tie(dtype, shape, offset, fortran_order) = parse_npy_header(fp);

  fclose(fp);
  return torch::Tensor();
}
}  // namespace cnpy