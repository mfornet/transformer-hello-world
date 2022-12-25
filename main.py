"""Main script for training and testing a transformer."""
from functools import partial
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim

from data import load_train_test_dataset, repeated
from model import Transformer


def accuracy(x, y, model):
    """Compute accuracy of a model."""
    with torch.no_grad():
        pred = model(x)
        pred = pred.argmax(dim=-1)
        acc = (pred == y)[:, 4:].float().mean()
        return acc.item()


def main(batch_size: int = 32, lr: float = 0.0001):
    """Train and evaluate a transformer model."""
    period = 3
    gn = partial(repeated, alpha="0123456789", period=period, length=15)

    x_train, y_train, x_test, y_test, alpha = load_train_test_dataset(gn)

    n_alpha = len(alpha)

    model = Transformer(
        len(alpha), d_model=32, n_heads=4, n_layers=4, d_ff=32, dropout=0.2
    )
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
            filter_pred = pred[:, period:, :]
            filter_y_batch = y_batch[:, period:]
            loss = F.nll_loss(
                filter_pred.contiguous().view(-1, n_alpha),
                filter_y_batch.contiguous().view(-1),
            )
            loss.backward()

            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1

        loss_mean = loss_sum / loss_count

        if epoch % 10 == 0:
            with torch.no_grad():
                pred = model(x_test)
                loss = F.nll_loss(
                    pred.view(-1, n_alpha), y_test.contiguous().view(-1)
                )

                acc_train = accuracy(x_train, y_train, model)
                acc_test = accuracy(x_test, y_test, model)

                print(
                    f"Epoch {epoch} loss: {loss_mean:3} test loss: {loss.item():3}"
                    f" train acc: {acc_train:3} test acc: {acc_test:3}"
                )

                # select one random element
                i = torch.randint(0, x_test.shape[0], (1,)).item()
                x = x_test[i].unsqueeze(0)
                pred = model(x)
                pred = pred.argmax(dim=-1)
                print("".join(alpha[i.item()] for i in x.squeeze(0)))
                print("_" + "".join(alpha[i.item()] for i in pred.squeeze(0)))
                print()


if __name__ == "__main__":
    main()
