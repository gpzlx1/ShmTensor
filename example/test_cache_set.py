import torch
import ShmTensorLib as capi
import numpy as np
from shmtensor import DiskTensor
import time


class time_recorder:

    def __init__(self, message="", output=True):
        self.message = message
        self.output = output

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        torch.cuda.synchronize()
        self.time = time.time() - self.start
        if self.output:
            print("{}\t\t\t{:.2f}".format(self.message, self.time * 1000))


def test_uva_set():

    # one dimension
    data = torch.randn(100_0000).float()
    capi.pin_memory(data)
    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel()).float().cuda()
        capi.uva_set(data, update_index, update_data)
        fetch = capi.uva_fetch(data, update_index)
        assert torch.equal(update_data, fetch)
    capi.unpin_memory(data)

    # mulitdimension
    data = torch.randn(100_0000, 128).float()
    capi.pin_memory(data)
    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel(), 128).float().cuda()
        capi.uva_set(data, update_index, update_data)
        fetch = capi.uva_fetch(data, update_index)
        assert torch.equal(update_data, fetch)
    capi.unpin_memory(data)


def test_cache_set():
    # one dimension
    data = torch.randn(100_0000).float()
    capi.pin_memory(data)

    cache_ratio = 0.7
    indices = torch.randperm(100_0000).long().cuda()
    indices = indices[:int(100_0000 * cache_ratio)]
    values = torch.arange(indices.numel()).long().cuda()
    gpu_data = data[indices.cpu()].cuda()
    hashmap = capi.CUCOStaticHashmap(indices, values, 0.8)

    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel()).float().cuda()
        capi.cache_set(data, gpu_data, update_index, update_data, hashmap)
        fetch = capi.cache_fetch(data, gpu_data, update_index, hashmap)
        assert torch.equal(update_data, fetch)
    capi.unpin_memory(data)

    # mulitdimension
    data = torch.randn(100_0000, 128).float()
    capi.pin_memory(data)

    cache_ratio = 0.7
    indices = torch.randperm(100_0000).long().cuda()
    indices = indices[:int(100_0000 * cache_ratio)]
    values = torch.arange(indices.numel()).long().cuda()
    gpu_data = data[indices.cpu()].cuda()
    hashmap = capi.CUCOStaticHashmap(indices, values, 0.8)

    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel(), 128).float().cuda()
        capi.cache_set(data, gpu_data, update_index, update_data, hashmap)
        fetch = capi.cache_fetch(data, gpu_data, update_index, hashmap)

        assert torch.equal(update_data, fetch)
    capi.unpin_memory(data)


def test_mask_set():
    # one dimension
    data = torch.randn(100_0000).float()
    capi.pin_memory(data)

    cache_ratio = 0.7
    indices = torch.randperm(100_0000).long().cuda()
    indices = indices[:int(100_0000 * cache_ratio)]
    values = torch.arange(indices.numel()).long().cuda()
    gpu_data = data[indices.cpu()].cuda()
    hashmap = capi.CUCOStaticHashmap(indices, values, 0.8)

    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel()).float().cuda()

        update_mask = hashmap.query(update_index)
        capi.cache_set_with_mask(data, gpu_data, update_index, update_data,
                                 update_mask)

        query_mask = hashmap.query(update_index)
        fetch = capi.cache_fetch_with_mask(data, gpu_data, update_index,
                                           query_mask)

        assert torch.equal(update_data, fetch)
    capi.unpin_memory(data)

    # multidimension
    data = torch.randn(100_0000, 128).float()
    capi.pin_memory(data)

    cache_ratio = 0.7
    indices = torch.randperm(100_0000).long().cuda()
    indices = indices[:int(100_0000 * cache_ratio)]
    values = torch.arange(indices.numel()).long().cuda()
    gpu_data = data[indices.cpu()].cuda()
    hashmap = capi.CUCOStaticHashmap(indices, values, 0.8)

    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel(), 128).float().cuda()

        update_mask = hashmap.query(update_index)
        capi.cache_set_with_mask(data, gpu_data, update_index, update_data,
                                 update_mask)

        query_mask = hashmap.query(update_index)
        fetch = capi.cache_fetch_with_mask(data, gpu_data, update_index,
                                           query_mask)

        assert torch.equal(update_data, fetch)
    capi.unpin_memory(data)


def benchmark():
    data = torch.randn(100_0000, 128).float()
    capi.pin_memory(data)
    full_gpu_data = data.cuda()

    cache_ratio = 1

    # create cache
    #indices = torch.arange(int(100_0000 * cache_ratio)).long().cuda()
    #indices = torch.randint(
    #    0, 100_0000, (int(100_0000 * cache_ratio), )).long().unique().cuda()
    indices = torch.randperm(100_0000).long().cuda()
    indices = indices[:int(100_0000 * cache_ratio)]
    print(indices)
    values = torch.arange(indices.numel()).long().cuda()
    gpu_data = data[indices.cpu()].cuda()
    hashmap = capi.CUCOStaticHashmap(indices, values, 0.8)

    for size in [10, 100, 1000, 10000, 100000, 30_0000, 100_0000]:
        update_index = torch.randint(0, 100_0000,
                                     (size, )).long().cuda().unique()
        update_data = torch.randn(update_index.numel(), 128).float().cuda()

        print("size", size)
        # benchmark
        for i in range(2):
            with time_recorder("cache_set", i != 0):
                capi.cache_set(data, gpu_data, update_index, update_data,
                               hashmap)

            with time_recorder("uva_set", i != 0):
                capi.uva_set(data, update_index, update_data)

            with time_recorder("cpu_set", i != 0):
                data[update_index.cpu()] = update_data.cpu()

            with time_recorder("gpu_set", i != 0):
                full_gpu_data[update_index] = update_data

            with time_recorder("mask_set", i != 0):
                mask = hashmap.query(update_index)
                capi.cache_set_with_mask(data, gpu_data, update_index,
                                         update_data, mask)
            print()
    capi.unpin_memory(data)


if __name__ == "__main__":
    test_uva_set()
    test_cache_set()
    test_mask_set()
    benchmark()
