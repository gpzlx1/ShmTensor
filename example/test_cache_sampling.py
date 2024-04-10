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
    print(sub_indptr)
    print(sub_indices)
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
    print(sub_indptr)
    print(sub_indices)
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


if __name__ == "__main__":
    test_csr_sampling()
