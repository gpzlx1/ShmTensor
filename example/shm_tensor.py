import torch
import torch.distributed as dist
import ShmTensorLib as capi
import weakref


class ShmTensor:

    def __init__(self,
                 name,
                 shape,
                 local_rank,
                 local_world_size,
                 local_group=None,
                 dtype=torch.float32,
                 pin_memory=False):
        self.name = name
        self.shape_ = shape
        self.dtype_ = dtype
        self.pin_memory_ = pin_memory
        self.cur_rank = local_rank
        self.cur_world_size = local_world_size
        self.cur_group = local_group

        self.size_ = self.dtype_.itemsize
        for i in shape:
            self.size_ *= i
        if self.size_ <= 0:
            raise Exception("Invalid shape")

        # check file name is not exist
        if capi.file_exist(name):
            raise Exception("File name is already exist")
        dist.barrier(self.cur_group)

        if self.cur_rank == 0:
            self.ptr_, self.fd_ = capi.create_shared_mem(
                self.name, self.size_, self.pin_memory_)
            dist.barrier(self.cur_group)
        else:
            dist.barrier(self.cur_group)
            self.ptr_, self.fd_ = capi.open_shared_mem(self.name, self.size_,
                                                       self.pin_memory_)

        # set finalizer
        # Note: It is important to ensure that func, args and kwargs do not own
        # any references to obj, either directly or indirectly, since otherwise
        # obj will never be garbage collected. In particular, func should not be
        # a bound method of obj
        self._shm_finalizer = weakref.finalize(self, self._cleanup_shm,
                                               self.name, self.size_,
                                               self.ptr_, self.fd_,
                                               self.pin_memory_)

        self.tensor_ = capi.open_shared_tensor(self.ptr_, self.dtype_,
                                               self.shape_)

    @classmethod
    def _cleanup_shm(self, name, size, ptr, fd, pin_memory):
        capi.release_shared_mem(name, size, ptr, fd, pin_memory)

    @property
    def shape(self):
        return self.tensor_.shape

    @property
    def dtype(self):
        return self.tensor_.dtype

    @property
    def device(self):
        return self.tensor_.device

    @property
    def size(self):
        return self.tensor_.size


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
