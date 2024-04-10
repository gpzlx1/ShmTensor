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


def test_uva_one_dim():
    data = torch.arange(100).float()
    capi.pin_memory(data)
    indices = torch.randint(0, 100, (10, )).long().cuda()
    output = capi.uva_fetch(data, indices)
    assert torch.equal(output, data.cuda()[indices])


def test_uva_multi_dim():
    data = torch.randn(100, 100, 100)
    capi.pin_memory(data)
    indices = torch.randint(0, 100, (10, )).long().cuda()
    output = capi.uva_fetch(data, indices)
    assert torch.equal(output, data.cuda()[indices])


def test_cache_multi_dim():
    data = torch.randn(100, 100, 100)
    capi.pin_memory(data)

    indices = torch.randint(0, 100, (50, )).long().unique().cuda()
    values = torch.arange(indices.numel()).long().cuda()
    gpu_data = data[indices.cpu()].cuda()
    hashmap = capi.CUCOStaticHashmap(indices, values, 0.8)

    query = torch.randint(0, 100, (20, )).long().cuda()
    output = capi.cache_fetch(data, gpu_data, query, hashmap)

    assert torch.equal(output, data[query.cpu()].cuda())


def benchmark():
    data = torch.randn(100_0000, 128).float()
    capi.pin_memory(data)
    full_gpu_data = data.cuda()

    cache_ratio = 0.9

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
        querys = torch.randint(0, 100_0000, (size, )).long().cuda()

        print("size", size)
        # benchmark
        for i in range(2):
            with time_recorder("cache_fetch", i != 0):
                a = capi.cache_fetch(data, gpu_data, querys, hashmap)

            with time_recorder("uva_fetch", i != 0):
                b = capi.uva_fetch(data, querys)

            with time_recorder("cpu_fetch", i != 0):
                c = data[querys.cpu()].cuda()

            with time_recorder("gpu_fetch", i != 0):
                d = full_gpu_data[querys]

            with time_recorder("mask_fetch", i != 0):
                mask = hashmap.query(querys)
                e = capi.cache_fetch_with_mask(data, gpu_data, querys, mask)
            print()

        assert torch.equal(a, b)
        assert torch.equal(a, c)
        assert torch.equal(a, d)
        assert torch.equal(a, e)
        # assert torch.equal(a, f)


if __name__ == "__main__":
    test_uva_one_dim()
    test_uva_multi_dim()
    test_cache_multi_dim()
    benchmark()
