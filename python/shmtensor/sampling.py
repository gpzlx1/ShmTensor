import torch
import torch.distributed as dist
import ShmTensorLib as capi
import weakref
import dgl
from dgl.heterograph import DGLBlock


def create_block_from_csc(indptr, indices, e_ids, num_src, num_dst):
    hgidx = dgl.heterograph_index.create_unitgraph_from_csr(
        2,
        num_src,
        num_dst,
        indptr,
        indices,
        e_ids,
        formats=['coo', 'csr', 'csc'],
        transpose=True)
    retg = DGLBlock(hgidx, (['_N'], ['_N']), ['_E'])
    return retg


def create_block_from_coo(row, col, num_src, num_dst):
    hgidx = dgl.heterograph_index.create_unitgraph_from_coo(
        2, num_src, num_dst, row, col, formats=['coo', 'csr', 'csc'])
    retg = DGLBlock(hgidx, (['_N'], ['_N']), ['_E'])
    return retg


class GPUSamplingDataloader:

    def __init__(self,
                 indptr,
                 indices,
                 seeds,
                 batchsize,
                 num_picks,
                 replace=False,
                 use_ddp=False,
                 shuffle=True,
                 drop_last=False):
        self.indptr = indptr
        self.indices = indices
        self.seeds = seeds
        self.batchsize = batchsize
        self.num_picks = num_picks
        self.replace = replace
        self.use_ddp = use_ddp
        self.shuffle = shuffle
        self.drop_last = drop_last

        if self.indptr.device == torch.device('cpu'):
            capi.pin_memory(self.indptr)
            weakref.finalize(self, capi.unpin_memory, self.indptr)

        if self.indices.device == torch.device('cpu'):
            capi.pin_memory(self.indices)
            weakref.finalize(self, capi.unpin_memory, self.indices)

        if self.use_ddp:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            numel = self.seeds.numel()
            partition_size = (numel + world_size - 1) // world_size
            self.seeds = self.seeds[rank * partition_size:(rank + 1) *
                                    partition_size]

        if self.shuffle:
            perm = torch.randperm(self.seeds.numel(), device=self.seeds.device)
            self.seeds = self.seeds[perm]

        if self.drop_last:
            self.len = self.seeds.numel() // self.batchsize
        else:
            self.len = (self.seeds.numel() + self.batchsize -
                        1) // self.batchsize
        self.curr = 0
        self.seeds = self.seeds.cuda()

    def __len__(self):
        return self.len

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.seeds.numel(), device=self.seeds.device)
            self.seeds = self.seeds[perm]

        self.curr = 0
        return self

    def __next__(self):
        if self.curr >= self.len:
            raise StopIteration

        seeds = self.seeds[self.curr * self.batchsize:(self.curr + 1) *
                           self.batchsize]
        self.curr += 1

        # begin sampling
        output_nodes = seeds
        result = []
        for num_pick in reversed(self.num_picks):
            sub_indptr, sub_indices = capi.csr_sampling(
                self.indptr, self.indices, seeds, num_pick, self.replace)
            unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
                [seeds, sub_indices])
            block = create_block_from_csc(sub_indptr, relabel_indices,
                                          torch.Tensor(),
                                          unique_tensor.numel(), seeds.numel())
            result.insert(0, block)
            seeds = unique_tensor

        intput_nodes = seeds

        return intput_nodes, output_nodes, result
