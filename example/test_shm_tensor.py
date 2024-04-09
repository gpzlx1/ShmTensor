import torch
import torch.distributed as dist
from shmtensor import ShmTensor

if __name__ == "__main__":
    # torch init process
    dist.init_process_group(backend='gloo', init_method='env://')

    #def test():
    tensor = ShmTensor("test", (2, 3, 4), dist.get_rank(),
                       dist.get_world_size(), None, torch.int16)
    print(tensor.tensor_)

    if dist.get_rank() == 0:
        tensor.tensor_[:] = 1000

    dist.barrier()

    print(tensor.tensor_)
