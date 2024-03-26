import torch
import ShmTensorLib as capi
import numpy as np
from disk_tensor import DiskTensor
from shm_tensor import ShmTensor
import torch.distributed as dist


def init_group():
    dist.init_process_group(backend='gloo', init_method='env://')


def test(shape, dtype):
    shm_tensor = ShmTensor("tmp", shape, dist.get_rank(),
                           dist.get_world_size(), None, dtype)
    shm_numpy = shm_tensor.tensor_.numpy()

    random_tensor = torch.rand_like(shm_tensor.tensor_,
                                    dtype=torch.float32).to(dtype)

    if dist.get_rank() == 0:
        shm_tensor.tensor_[:] = random_tensor

        # save to disk
        np.save("tmp.npy", shm_numpy)

    dist.barrier()
    del shm_numpy
    del shm_tensor

    # disk_tensor load
    disk_tensor = DiskTensor("tmp.npy")

    assert disk_tensor.shape == random_tensor.shape
    assert disk_tensor.dtype == random_tensor.dtype
    assert torch.equal(disk_tensor.tensor_, random_tensor)


init_group()

for shape in [
    (10, ),
    (10, 20),
    (10, 20, 30),
    (10, 20, 30, 40),
    (100_0000, 128),
]:
    for dtype in [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
    ]:
        print(test.__name__, shape, dtype)
        test(shape, dtype)
