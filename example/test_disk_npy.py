import torch
import ShmTensorLib as capi
import numpy as np

# capi.load_npy("a.npy")


def test(shape, dtype):
    npy_data = np.random.randn(*shape).astype(dtype)
    npy_tensor = torch.from_numpy(npy_data)

    # save
    np.save("a.npy", npy_data)

    # load
    th_tensor, size, ptr, fd = capi.open_mmap_tensor("a.npy", False)

    # check
    assert th_tensor.size() == npy_tensor.size()
    assert th_tensor.dtype == npy_tensor.dtype
    assert th_tensor.equal(npy_tensor)

    del th_tensor
    capi.close_mmap_tensor(size, ptr, fd, False)


for shape in [
    (10, ),
    (10, 20),
    (10, 20, 30),
    (10, 20, 30, 40),
    (100_0000, 128),
]:
    for dtype in [
            np.uint8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
    ]:
        print(test.__name__, shape, dtype)
        test(shape, dtype)
