import torch
import ShmTensorLib
import time
import numpy as np

keys = torch.randint(0, 1000_0000, (100_0000, )).cuda().int().unique()
values = torch.range(0, keys.numel()).cuda().int()

cuco = ShmTensorLib.CUCOStaticHashmap(keys, values, 0.5)

requests = torch.randint(0, 1000_0000, (100_0000, )).cuda().int()


def bench(map, requests):
    time_list = []
    print("---")
    for i in range(10):
        torch.cuda.synchronize()
        begin = time.time()
        map.query(requests)
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)
        # assert(keys.eq(requests).all())
    print(np.mean(time_list[2:]) * 1000)
    print("---\n")


bench(map=cuco, requests=requests)

for load_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]:
    cuco = ShmTensorLib.CUCOStaticHashmap(keys, values, load_factor)
    print(f"Load factor: {load_factor}")
    bench(map=cuco, requests=requests)
