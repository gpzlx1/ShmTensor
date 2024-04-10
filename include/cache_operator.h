#pragma once
#include <common.h>
#include <torch/extension.h>
#include "cuco_hashmap.h"

torch::Tensor CacheTensorFetch(torch::Tensor uva_data, torch::Tensor gpu_data,
                               torch::Tensor indices,
                               pycuco::CUCOHashmapWrapper& hashmap);

torch::Tensor CacheTensorFetchWithMask(torch::Tensor uva_data,
                                       torch::Tensor gpu_data,
                                       torch::Tensor indices,
                                       torch::Tensor mask);

torch::Tensor UVATensorFetch(torch::Tensor& uva_data, torch::Tensor& indices);

void CacheTensorSet(torch::Tensor uva_data, torch::Tensor gpu_data,
                    torch::Tensor indices, torch::Tensor data,
                    pycuco::CUCOHashmapWrapper& hashmap);

void CacheTensorSetWithMask(torch::Tensor uva_data, torch::Tensor gpu_data,
                            torch::Tensor indices, torch::Tensor data,
                            torch::Tensor mask);
void UVATensorSet(torch::Tensor& uva_data, torch::Tensor& indices,
                  torch::Tensor& data);

void PinMemory(torch::Tensor& data);

void UnpinMemory(torch::Tensor& data);