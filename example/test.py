import torch
import ShmTensorLib as capi
import numpy as np
from disk_tensor import DiskTensor

# save
npy_data = np.random.randn(100, 100, 100, 100).astype(np.int32)
npy_tensor = torch.from_numpy(npy_data)
np.save("a.npy", npy_data)

# load
disk_tensor = DiskTensor("a.npy", True)

print(disk_tensor.tensor_)

capi.thrust_scan(disk_tensor.tensor_)

print(disk_tensor.tensor_)
