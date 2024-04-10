import torch
import ShmTensorLib as capi
import numpy as np
from shmtensor import DiskTensor

data = torch.arange(100).float()
capi.pin_memory(data)

indices = torch.randint(0, 100, (10, )).long().cuda()

output = capi.uva_fetch(data, indices)

print(output)
assert torch.equal(output, data.cuda()[indices])

data = torch.randn(100, 100, 100)
capi.pin_memory(data)

indices = torch.randint(0, 100, (10, )).long().cuda()

print(data)
print(indices)

output = capi.uva_fetch(data, indices)

print(output)
assert torch.equal(output, data.cuda()[indices])
'''
# save
npy_data = np.random.randn(100, 100, 100, 100).astype(np.int32)
npy_tensor = torch.from_numpy(npy_data)
np.save("a.npy", npy_data)

# load
disk_tensor = DiskTensor("a.npy", True)

print(disk_tensor.tensor_)

capi.thrust_scan(disk_tensor.tensor_)

print(disk_tensor.tensor_)
'''
