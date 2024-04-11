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

std::tuple<torch::Tensor, torch::Tensor> CSRWiseSampling(torch::Tensor indptr,
                                                         torch::Tensor indices,
                                                         torch::Tensor seeds,
                                                         int64_t num_picks,
                                                         bool replace);

std::tuple<torch::Tensor, torch::Tensor> CSRWiseCacheSampling(
    torch::Tensor uva_indptr, torch::Tensor uva_indices,
    torch::Tensor gpu_indptr, torch::Tensor gpu_indices,
    pycuco::CUCOHashmapWrapper& hashmap, torch::Tensor seeds, int64_t num_picks,
    bool replace);

std::tuple<torch::Tensor, torch::Tensor> CreateCacheCSR(
    torch::Tensor uva_indptr, torch::Tensor uva_indices, torch::Tensor seeds);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabel(
    std::vector<torch::Tensor> tensors);