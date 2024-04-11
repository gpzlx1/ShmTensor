from shmtensor import GPUSamplingDataloader
from dgl.data import RedditDataset
import torch
import dgl
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


dgl_graph = RedditDataset()[0]
indptr, indices, _ = dgl_graph.adj_tensors('csc')
train_nids = torch.nonzero(dgl_graph.ndata['train_mask']).squeeze(1)

dataloader = GPUSamplingDataloader(indptr.clone(),
                                   indices.clone(),
                                   train_nids.clone(),
                                   batchsize=1000,
                                   num_picks=[10, 25],
                                   shuffle=True)

for _ in range(2):
    with time_recorder("ours_uva", _ != 0):
        for step, i in enumerate(dataloader):
            pass

dataloader = GPUSamplingDataloader(indptr.clone().cuda(),
                                   indices.clone().cuda(),
                                   train_nids.clone().cuda(),
                                   batchsize=1000,
                                   num_picks=[10, 25],
                                   shuffle=True)

for _ in range(2):
    with time_recorder("ours_gpu", _ != 0):
        for step, i in enumerate(dataloader):
            pass

# dgl sampling
sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 25])
dgl_dataloader = dgl.dataloading.DataLoader(dgl_graph,
                                            train_nids,
                                            sampler,
                                            batch_size=1000,
                                            use_uva=True,
                                            shuffle=True)

for _ in range(2):
    with time_recorder("dgl_uva", _ != 0):
        for step, j in enumerate(dgl_dataloader):
            pass

# dgl sampling
sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 25])
dgl_dataloader = dgl.dataloading.DataLoader(dgl_graph.to('cuda'),
                                            train_nids.cuda(),
                                            sampler,
                                            batch_size=1000,
                                            use_uva=False,
                                            shuffle=True)

for _ in range(2):
    with time_recorder("dgl_gpu", _ != 0):
        for step, j in enumerate(dgl_dataloader):
            pass
