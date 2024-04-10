import torch
import ShmTensorLib

torch.manual_seed(32)

keys = torch.randint(0, 10000, (100, )).int().cuda().unique()
values = keys + 1
print(keys)
print(values)

# Test 1
keys = keys.int()
values = values.int()
hashmap = ShmTensorLib.CUCOStaticHashmap(keys, values, 0.5)
print()
print(hashmap)
print(hashmap.query(keys.long() + 1))

# Test 2
keys = keys.long()
values = values.int()
hashmap = ShmTensorLib.CUCOStaticHashmap(keys, values, 0.5)
print()
print(hashmap)
print(hashmap.query(keys.long() + 1))

# Test 3
keys = keys.int()
values = values.long()
hashmap = ShmTensorLib.CUCOStaticHashmap(keys, values, 0.5)
print()
print(hashmap)
print(hashmap.query(keys.long() + 1))

# Test 4
keys = keys.long()
values = values.long()
hashmap = ShmTensorLib.CUCOStaticHashmap(keys, values, 0.5)
print()
print(hashmap)
print(hashmap.query(keys.long() + 1))


# Test 5
print()
print(hashmap.capacity())
print(hashmap.memory_usage())
