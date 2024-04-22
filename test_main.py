import unittest
import torch
from main import tensor_to_str


def test_tensor_to_str():
    x = torch.tensor(
        [1.0],
        dtype=torch.float32,
    )
    assert tensor_to_str(x) == 'Tensor(shape=(1,), min=1.0, max=1.0, mean=1.0, dtype=torch.float32, device=cpu)'

    x = torch.tensor(
        [1.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
    ).reshape(2, 2)
    assert tensor_to_str(x) == 'Tensor(shape=(2, 2), min=1.0, max=1.0, mean=1.0, std=0.0, dtype=torch.float32, device=cpu)'

    x = torch.tensor(
        [1.0, 3.0, 5.0],
        dtype=torch.float32,
    )
    assert tensor_to_str(x) == 'Tensor(shape=(3,), min=1.0, max=5.0, mean=3.0, std=2.0, dtype=torch.float32, device=cpu)'
    print(tensor_to_str(x))

    x = torch.tensor(
        [1.0, float('nan'), 5.0],
        dtype=torch.float32,
    )
    assert tensor_to_str(x) == 'Tensor(shape=(3,), n_nans=1, dtype=torch.float32, device=cpu)'

    x = torch.tensor(
        [1.0, float('+inf'), 5.0],
        dtype=torch.float32,
    )
    assert tensor_to_str(x) == 'Tensor(shape=(3,), n_infs=1, dtype=torch.float32, device=cpu)'

    x = torch.tensor(
        [1, 2, 3, 4],
        dtype=torch.int32,
    )
    assert tensor_to_str(x) == 'Tensor(shape=(4,), min=1, max=4, sum=10, dtype=torch.int32, device=cpu)'

    x = torch.tensor(
        [False, True],
        dtype=torch.bool,
    )
    assert tensor_to_str(x) == 'Tensor(shape=(2,), sum=1, dtype=torch.bool, device=cpu)'


if __name__ == '__main__':
    unittest.main()
