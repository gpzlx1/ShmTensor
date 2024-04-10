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

    dgl_graph = RedditDataset()[0]
    indptr, indices, _ = dgl_graph.adj_tensors('csc')
    full_gpu_indptr = indptr.cuda()
    full_gpu_indices = indices.cuda()

    capi.pin_memory(indptr)
    capi.pin_memory(indices)

    for size in [10, 100, 1000, 10000, 10_0000]:
        seeds = torch.randperm(dgl_graph.num_nodes())[:size].cuda()
        print(seeds.numel())

        for i in range(2):

            with time_recorder("gpu", i != 0):
                ## wise sampling
                sub_indptr, sub_indices = capi.csr_sampling(
                    full_gpu_indptr, full_gpu_indices, seeds, 5, False)
                ## tensor relabel
                unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
                    [seeds, sub_indices])
                ## create block
                create_block(
                    ('csc', (sub_indptr, relabel_indices, torch.Tensor())),
                    num_src_nodes=unique_tensor.numel(),
                    num_dst_nodes=seeds.numel(),
                    device='cuda')

            with time_recorder("uva", i != 0):
                ## wise sampling
                sub_indptr, sub_indices = capi.csr_sampling(
                    indptr, indices, seeds, 5, False)
                ## tensor relabel
                unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
                    [seeds, sub_indices])
                ## create block
                create_block(
                    ('csc', (sub_indptr, relabel_indices, torch.Tensor())),
                    num_src_nodes=unique_tensor.numel(),
                    num_dst_nodes=seeds.numel(),
                    device='cuda')

    capi.unpin_memory(indptr)
    capi.unpin_memory(indices)


if __name__ == "__main__":
    test_csr_sampling()
    test_tensor_relabel()
    test_dgl_sampling()
    benchmark()
