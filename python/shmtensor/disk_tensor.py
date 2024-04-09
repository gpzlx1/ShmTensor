import torch
import torch.distributed as dist
import ShmTensorLib as capi
import weakref
import os


class DiskTensor:

    def __init__(self, filename, pin_memory=False):
        self.filename_ = filename
        self.pin_memory_ = pin_memory

        # check file exists and file is .npy file
        assert os.path.exists(filename)
        assert filename.endswith('.npy')

        # open file
        self.tensor_, self.size_, self.ptr_, self.fd_ = capi.open_mmap_tensor(
            filename, self.pin_memory_)

        # set finalizer
        self._disk_finalizer = weakref.finalize(self, self._cleanup_mmap,
                                                self.tensor_, self.size_,
                                                self.ptr_, self.fd_,
                                                self.pin_memory_)

    @classmethod
    def _cleanup_mmap(cls, tensor, size, ptr, fd, pin_memory):
        del tensor
        capi.close_mmap_tensor(size, ptr, fd, pin_memory)

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
