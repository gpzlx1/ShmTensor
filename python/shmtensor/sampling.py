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
            if getattr(self, 'has_cache', None):
                sub_indptr, sub_indices = capi.csr_cache_sampling(
                    self.indptr, self.indices, self.gpu_indptr,
                    self.gpu_indices, self.hashmap, seeds, num_pick,
                    self.replace)
            else:
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

    def create_cache(self, cache_capacity, hotness):
        if cache_capacity <= 0:
            return

        # test wheather full cache
        full_size = self.indptr.nbytes + self.indices.nbytes

        if full_size <= cache_capacity:
            self.indices = self.indices.cuda()
            self.indptr = self.indptr.cuda()
            print("Cache Ratio for GPU sampling: {:.2f}".format(
                cache_capacity / full_size))
            return

        degress = self.indptr[1:] - self.indptr[:-1]
        _, cache_candidates = torch.sort(hotness, descending=True)

        # compute size
        size = degress[cache_candidates] * self.indices.element_size(
        ) + self.indptr.element_size()
        prefix_sum_size = torch.cumsum(size, dim=0)
        cache_size = torch.searchsorted(prefix_sum_size, cache_capacity).item()
        cache_candidates = cache_candidates[:cache_size].cuda()

        # binary search
        if cache_candidates.numel() > 0:
            self.gpu_indptr, self.gpu_indices = capi.create_subcsr(
                self.indptr, self.indices, cache_candidates)
            self.hashmap = capi.CUCOStaticHashmap(
                cache_candidates,
                torch.arange(cache_candidates.numel(),
                             device='cuda',
                             dtype=cache_candidates.dtype), 0.8)
            self.has_cache = True
            print("Cache Ratio for GPU sampling: {:.2f}".format(
                prefix_sum_size[cache_size].item() / full_size))
            print("create cache success")

    def presampling(self):
        sampling_hotness = torch.zeros(self.indptr.numel() - 1, device='cpu')
        feature_hotness = torch.zeros(self.indptr.numel() - 1, device='cpu')

        for i in range(self.len):
            seeds = self.seeds[i * self.batchsize:(i + 1) * self.batchsize]

            for num_pick in reversed(self.num_picks):
                sampling_hotness[seeds.cpu()] += 1

                sub_indptr, sub_indices = capi.csr_sampling(
                    self.indptr, self.indices, seeds, num_pick, self.replace)
                unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
                    [seeds, sub_indices])
                seeds = unique_tensor

                feature_hotness[unique_tensor.cpu()] += 1

        if self.use_ddp:
            sampling_hotness = sampling_hotness.cuda()
            feature_hotness = feature_hotness.cuda()

            torch.distributed.all_reduce(sampling_hotness)
            torch.distributed.all_reduce(feature_hotness)

            sampling_hotness = sampling_hotness.cpu()
            feature_hotness = feature_hotness.cpu()

        return sampling_hotness, feature_hotness
