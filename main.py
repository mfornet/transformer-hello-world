"""Main script for training and testing a transformer."""
from functools import partial
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim

from data import load_train_test_dataset, repeated
from model import Transformer


def main(epochs: int = 100, batch_size: int = 32, lr: float = 0.01):
    """Train and evaluate a transformer model."""
    gn = partial(repeated, alpha="0123456789", period=4, length=15)
    x_train, y_train, x_test, y_test, alpha = load_train_test_dataset(gn)

    n_alpha = len(alpha)

    model = Transformer(len(alpha), d_model=16, n_heads=8, n_layers=2, d_ff=16)
    model.requires_grad_()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in count(1):
        n = x_train.shape[0]

        perm = torch.randperm(n)
        x_train = x_train[perm]
        y_train = y_train[perm]

        loss_sum = 0.0
        loss_count = 0

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            model.zero_grad()

            pred = model(x_batch)
            loss = F.nll_loss(pred.view(-1, n_alpha), y_batch.view(-1))
            loss.backward()

            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1

        loss_mean = loss_sum / loss_count

        if epoch % 5 == 0:
            with torch.no_grad():
                pred = model(x_test)
                loss = F.nll_loss(
                    pred.view(-1, n_alpha), y_test.contiguous().view(-1)
                )

                print(
                    f"Epoch {epoch} loss: {loss_mean} test loss: {loss.item()}"
                )


if __name__ == "__main__":
    main()
