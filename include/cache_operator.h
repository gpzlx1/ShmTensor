#pragma once
#include <common.h>
#include <torch/extension.h>
#include "cuco_hashmap.h"

torch::Tensor CacheTensorFetch(torch::Tensor uva_data, torch::Tensor gpu_data,
                               torch::Tensor indices,
                               pycuco::CUCOHashmapWrapper& hashmap);

torch::Tensor UVATensorFetch(torch::Tensor& uva_data, torch::Tensor& indices);

void PinMemory(torch::Tensor& data);

void UnpinMemory(torch::Tensor& data);