import torch
import ShmTensorLib as capi
import numpy as np
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


def test_csr_sampling():
    num_picks = 5
    indptr = torch.tensor([0, 0, 4, 9, 15, 100, 101, 107]).long().cuda()
    indices = torch.arange(107).long().cuda()

    seeds = torch.tensor([5, 0, 1, 2, 3, 4, 6]).int().cuda()

    replace = True
    sub_indptr, sub_indices = capi.csr_sampling(indptr, indices, seeds,
                                                num_picks, replace)
    #print(sub_indptr)
    #print(sub_indices)
    assert torch.equal(
        sub_indptr, torch.tensor([0, 5, 5, 10, 15, 20, 25, 30], device='cuda'))
    assert sub_indices.numel() == 30

    assert ((sub_indices[0:5] >= 100) & (sub_indices[0:5] < 101)).any()
    assert ((sub_indices[5:10] >= 0) & (sub_indices[5:10] < 4)).any()
    assert ((sub_indices[10:15] >= 4) & (sub_indices[10:15] < 9)).any()
    assert ((sub_indices[15:20] >= 9) & (sub_indices[15:20] < 15)).any()
    assert ((sub_indices[20:25] >= 15) & (sub_indices[20:25] < 100)).any()
    assert ((sub_indices[25:30] >= 101) & (sub_indices[25:30] < 107)).any()

    replace = False
    sub_indptr, sub_indices = capi.csr_sampling(indptr, indices, seeds,
                                                num_picks, replace)
    #print(sub_indptr)
    #print(sub_indices)
    assert torch.equal(
        sub_indptr, torch.tensor([0, 1, 1, 5, 10, 15, 20, 25], device='cuda'))
    assert sub_indices.numel() == 25
    assert sub_indices[0] == 100
    assert torch.equal(sub_indices[1:5],
                       torch.tensor([0, 1, 2, 3], device='cuda'))
    assert torch.equal(sub_indices[5:10],
                       torch.tensor([4, 5, 6, 7, 8], device='cuda'))
    assert torch.equal(sub_indices[10:15],
                       torch.tensor([9, 10, 11, 12, 13], device='cuda'))
    assert ((sub_indices[15:20] >= 15) & (sub_indices[15:20] < 100)).any()
    assert sub_indices[15:20].unique().numel() == 5
    assert ((sub_indices[20:25] >= 101) & (sub_indices[20:25] < 107)).any()
    assert sub_indices[20:25].unique().numel() == 5


def test_csr_cache_sampling():
    num_picks = 5
    indptr = torch.tensor([0, 0, 4, 9, 15, 100, 101, 107]).long()
    indices = torch.arange(107).long()
    capi.pin_memory(indptr)
    capi.pin_memory(indices)

    # create gpu cache
    cache_seeds = torch.tensor([5, 2, 1]).long().cuda()
    gpu_indptr, gpu_indices = capi.create_subcsr(indptr, indices, cache_seeds)
    hashmap = capi.CUCOStaticHashmap(
        cache_seeds,
        torch.arange(cache_seeds.numel()).cuda().long(), 0.8)

    assert torch.equal(gpu_indptr, torch.tensor([0, 1, 6, 10], device='cuda'))
    assert torch.equal(
        gpu_indices,
        torch.tensor([100, 4, 5, 6, 7, 8, 0, 1, 2, 3], device='cuda'))

    # test sampling
    seeds = torch.tensor([5, 0, 1, 2, 3, 4, 6]).int().cuda()

    replace = True
    sub_indptr, sub_indices = capi.csr_cache_sampling(indptr, indices,
                                                      gpu_indptr, gpu_indices,
                                                      hashmap, seeds,
                                                      num_picks, replace)
    assert torch.equal(
        sub_indptr, torch.tensor([0, 5, 5, 10, 15, 20, 25, 30], device='cuda'))
    assert sub_indices.numel() == 30

    assert ((sub_indices[0:5] >= 100) & (sub_indices[0:5] < 101)).any()
    assert ((sub_indices[5:10] >= 0) & (sub_indices[5:10] < 4)).any()
    assert ((sub_indices[10:15] >= 4) & (sub_indices[10:15] < 9)).any()
    assert ((sub_indices[15:20] >= 9) & (sub_indices[15:20] < 15)).any()
    assert ((sub_indices[20:25] >= 15) & (sub_indices[20:25] < 100)).any()
    assert ((sub_indices[25:30] >= 101) & (sub_indices[25:30] < 107)).any()

    replace = False
    sub_indptr, sub_indices = capi.csr_cache_sampling(indptr, indices,
                                                      gpu_indptr, gpu_indices,
                                                      hashmap, seeds,
                                                      num_picks, replace)
    assert torch.equal(
        sub_indptr, torch.tensor([0, 1, 1, 5, 10, 15, 20, 25], device='cuda'))
    assert sub_indices.numel() == 25
    assert sub_indices[0] == 100
    assert torch.equal(sub_indices[1:5],
                       torch.tensor([0, 1, 2, 3], device='cuda'))
    assert torch.equal(sub_indices[5:10],
                       torch.tensor([4, 5, 6, 7, 8], device='cuda'))
    assert torch.equal(sub_indices[10:15],
                       torch.tensor([9, 10, 11, 12, 13], device='cuda'))
    assert ((sub_indices[15:20] >= 15) & (sub_indices[15:20] < 100)).any()
    assert sub_indices[15:20].unique().numel() == 5
    assert ((sub_indices[20:25] >= 101) & (sub_indices[20:25] < 107)).any()
    assert sub_indices[20:25].unique().numel() == 5

    capi.unpin_memory(indptr)
    capi.unpin_memory(indices)


def test_tensor_relabel():
    data1 = torch.tensor([2, 0, 4, 1, 4, 0, 4, 1, 6, 4]).cuda()
    data2 = torch.tensor([3, 8, 5, 10, 10, 6, 11, 11, 9, 14]).cuda()
    unique_tensor, relabel_tensors = capi.tensor_relabel([data1, data2])

    assert torch.equal(
        unique_tensor,
        torch.tensor([2, 0, 4, 1, 6, 3, 8, 5, 10, 11, 9, 14]).cuda())
    assert torch.equal(relabel_tensors[0],
                       torch.tensor([0, 1, 2, 3, 2, 1, 2, 3, 4, 2]).cuda())
    assert torch.equal(relabel_tensors[1],
                       torch.tensor([5, 6, 7, 8, 8, 4, 9, 9, 10, 11]).cuda())


def test_dgl_sampling():
    from dgl import create_block
    from dgl.data import RedditDataset

    dgl_graph = RedditDataset()[0]
    indptr, indices, _ = dgl_graph.adj_tensors('csc')

    # sampling
    seeds = torch.randperm(dgl_graph.num_nodes())[:1000].cuda()
    ## wise sampling
    sub_indptr, sub_indices = capi.csr_sampling(indptr.cuda(), indices.cuda(),
                                                seeds, 5, False)
    ## tensor relabel
    unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
        [seeds, sub_indices])
    ## create block
    block = create_block(
        ('csc', (sub_indptr, relabel_indices, torch.Tensor())),
        num_src_nodes=unique_tensor.numel(),
        num_dst_nodes=seeds.numel(),
        device='cuda')
    print(block)


def benchmark():
    from dgl import create_block
    from dgl.data import RedditDataset

    # dgl_graph = RedditDataset()[0]
    indptr = torch.randint(0, 1000, (200_0000, )).long()
    indptr = torch.cumsum(indptr, dim=0)
    indptr = torch.cat((torch.tensor([0], device='cpu',
                                     dtype=torch.int64), indptr))
    indices = torch.arange(indptr[-1].item()).long()

    full_gpu_indptr, full_gpu_indices = indptr.cuda(), indices.cuda()

    capi.pin_memory(indptr)
    capi.pin_memory(indices)

    # create gpu_indptr, gpu_indices
    cache_ratio = 0.3
    ## select max degree node
    degree = indptr[1:] - indptr[:-1]
    _, seeds = torch.sort(degree, descending=True)
    seeds = seeds[:int(degree.numel() * cache_ratio)].cuda()
    gpu_indptr, gpu_indices = capi.create_subcsr(indptr, indices, seeds)
    hashmap = capi.CUCOStaticHashmap(seeds,
                                     torch.arange(seeds.numel()).cuda().long(),
                                     0.8)

    for size in [10, 100, 1000, 10000, 10_0000, 30_0000, 50_0000]:
        seeds = torch.randperm(200_0000)[:size].cuda()
        print()
        print(seeds.numel())

        for j in range(2):

            with time_recorder("gpu", j != 0):
                ## wise sampling
                for i in range(50):
                    sub_indptr, sub_indices = capi.csr_sampling(
                        full_gpu_indptr, full_gpu_indices, seeds, 5, False)

            with time_recorder("uva", j != 0):
                ## wise sampling
                for i in range(50):
                    sub_indptr, sub_indices = capi.csr_sampling(
                        indptr, indices, seeds, 5, False)

            with time_recorder("cache", j != 0):
                ## wise sampling
                for i in range(50):
                    sub_indptr, sub_indices = capi.csr_cache_sampling(
                        indptr, indices, gpu_indptr, gpu_indices, hashmap,
                        seeds, 5, False)

    capi.unpin_memory(indptr)
    capi.unpin_memory(indices)


if __name__ == "__main__":
    test_csr_sampling()
    test_csr_cache_sampling()
    test_tensor_relabel()
    test_dgl_sampling()
    benchmark()
