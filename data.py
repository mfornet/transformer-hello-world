"""Data utilities."""
from typing import Set, Iterable, Tuple
from itertools import product

import torch


def dataset_into_tensor(data: list[str], alpha: str) -> torch.Tensor:
    """Convert a dataset into a tensor."""
    assert alpha[0] == "\0", "The null character is reserved for padding."
    assert len(alpha) == len(set(alpha)), "The alphabet must be unique."

    n = len(data)
    m = max(len(s) for s in data) + 1
    tensor = torch.zeros(n, m, dtype=torch.int64)
    for i, seq in enumerate(data):
        for j, char in enumerate(seq):
            tensor[i, j] = alpha.index(char)

    return tensor


def tensor_into_dataset(data: torch.Tensor, alpha: str, truncate: bool = True):
    """Convert a tensor into a dataset."""
    assert alpha[0] == "\0", "The null character is reserved for padding."
    assert len(alpha) == len(set(alpha)), "The alphabet must be unique."
    assert data.dtype == torch.int64, "The tensor must be of type int64."

    n, m, _ = data.shape
    dataset = []
    for i in range(n):
        s = ""
        for j in range(m):
            if truncate and data[i, j] == 0:
                break
            s += alpha[data[i, j]]
        dataset.append(s)
    return dataset


def alpha_from_dataset(data: list[str]) -> str:
    """Create an alphabet from a dataset."""
    alpha: Set[str] = set()
    for seq in data:
        alpha = alpha.union(set(seq))
    return "\0" + "".join(sorted(alpha))


def load_dataset(gn) -> Tuple[torch.Tensor, str]:
    """Load dataset from generator."""
    data = list(gn())
    alpha = alpha_from_dataset(data)
    tensor = dataset_into_tensor(data, alpha)
    return tensor, alpha


def load_train_test_dataset(
    gn, ratio: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Load train and test dataset from generator."""
    data, alpha = load_dataset(gn)
    n = data.shape[0]
    torch.random.manual_seed(0)

    data = data[torch.randperm(n)]
    n_train = int(n * ratio)

    x_train = data[:n_train, :-1]
    y_train = data[:n_train, 1:]

    x_test = data[n_train:, :-1]
    y_test = data[n_train:, 1:]

    return x_train, y_train, x_test, y_test, alpha


def repeated(alpha: str, period: int, length: int) -> Iterable[str]:
    """Generate all possible repeated strings of a given length."""
    times = (length + period - 1) // period
    for prefix_t in product(alpha, repeat=period):
        prefix = "".join(prefix_t)
        prefix *= times
        yield prefix[:length]
