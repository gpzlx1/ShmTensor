import torch
import ShmTensorLib as capi

one = torch.ones(10).int().cuda()

capi.thrust_scan(one)

print(one)
