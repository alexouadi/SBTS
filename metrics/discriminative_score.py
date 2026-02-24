import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


def train_test_divide(data_x, data_x_hat, train_rate=0.8):
    """Split real and synthetic datasets into train and test partitions.

    Parameters
    ----------
    data_x : np.ndarray | torch.Tensor
        Real sequences of shape ``(batch, seq_len, dim)``.
    data_x_hat : np.ndarray | torch.Tensor
        Synthetic sequences of shape ``(batch, seq_len, dim)``.
    train_rate : float, optional
        Fraction of samples assigned to the training split.

    Returns
    -------
    tuple
        ``(train_x, train_x_hat, test_x, test_x_hat)``.
    """
    # Divide train / test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = data_x[train_idx, :, :]
    test_x = data_x[test_idx, :, :]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = data_x_hat[train_idx, :, :]
    test_x_hat = data_x_hat[test_idx, :, :]

    return train_x, train_x_hat, test_x, test_x_hat


def batch_generator(data, batch_size):
    """Sample a random mini-batch from a sequence dataset.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input sequences.
    batch_size : int
        Number of sampled trajectories.

    Returns
    -------
    np.ndarray | torch.Tensor
        Batch with shape ``(batch_size, seq_len, dim)``.
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = data[train_idx, :, :]

    return X_mb


class Discriminator(nn.Module):
    """GRU-based discriminator used in the discriminative score metric."""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


def discriminative_score_metrics(ori_data, generated_data, iterations, device=torch.device('cpu'), device_ids=[2]):
    """Compute the discriminative score between real and synthetic data.

    Parameters
    ----------
    ori_data : torch.Tensor
        Real sequences with shape ``(batch, seq_len, dim)``.
    generated_data : torch.Tensor
        Synthetic sequences with shape ``(batch, seq_len, dim)``.
    iterations : int
        Number of discriminator optimization steps.
    device : torch.device, optional
        Device used for training and inference.
    device_ids : list[int], optional
        GPU IDs used by ``torch.nn.DataParallel``.

    Returns
    -------
    float
        Absolute distance between discriminator accuracy and random-guess level (0.5).
    """
    ori_data = ori_data.to(device)
    generated_data = generated_data.to(device)

    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Build a post-hoc RNN discriminator network
    hidden_dim = int(dim/2)
    batch_size = 128
    num_layers = 2

    discriminator = Discriminator(dim, hidden_dim, num_layers)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)
    discriminator = discriminator.to(device)

    d_optimizer = optim.Adam(discriminator.parameters())

    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)
    losses = []

    # Training step
    for itt in range(iterations):
        X_mb = batch_generator(train_x, batch_size)
        X_hat_mb = batch_generator(train_x_hat, batch_size)

        d_optimizer.zero_grad()

        y_logit_real, _ = discriminator(X_mb)
        y_logit_fake, _ = discriminator(X_hat_mb)

        d_loss_real = nn.BCEWithLogitsLoss()(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = nn.BCEWithLogitsLoss()(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        d_optimizer.step()

        # Print the training loss over the epoch.
        losses.append(d_loss.item())

    # Test the performance on the testing set
    _, y_pred_real_curr = discriminator(test_x)
    _, y_pred_fake_curr = discriminator(test_x_hat)

    y_pred_final = np.squeeze(
        np.concatenate((y_pred_real_curr.detach().cpu().numpy(), y_pred_fake_curr.detach().cpu().numpy()), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
