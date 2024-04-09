import torch
from shmtensor import DiskTensor

if __name__ == "__main__":
    import numpy as np
    npy_data = np.random.randn(2, 3, 4).astype(np.float64)
    npy_tensor = torch.from_numpy(npy_data)

    # save
    np.save("a.npy", npy_data)

    tensor = DiskTensor("a.npy")

    assert torch.equal(tensor.tensor_, npy_tensor)

    print(tensor.tensor_)
    print(npy_tensor)
