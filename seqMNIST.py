# Based on an example on the PyTorch Lightning homepage:
# https://github.com/williamFalcon/pytorch-lightning#how-do-i-do-use-it

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl


class SeqMNIST(pl.LightningModule):

    def __init__(self, model, learning_rate, default_batch_size, is_permuted, percent_validation=0.25):
        super(SeqMNIST, self).__init__()
        self.mnist_dim = 28 * 28
        self.model = model
        self.learning_rate = learning_rate
        self.default_batch_size = default_batch_size
        self.fixed_permutation = None
        if is_permuted:
            print('Running permuted version')
            self.fixed_permutation = torch.randperm(self.mnist_dim)

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        assert 0 <= percent_validation < 1
        val_nb = round(len(full_train_dataset) * percent_validation)
        tng_nb = len(full_train_dataset) - val_nb
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_train_dataset, [tng_nb, val_nb])
        self.test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        assert len(self.train_dataset) + len(self.val_dataset) == len(full_train_dataset)
        print(f'training samples:   {tng_nb:>7}')
        print(f'validation samples: {val_nb:>7}')
        print(f'test samples:       {len(self.test_dataset):>7}')
        print('')

    def forward(self, x):
        if self.fixed_permutation is not None:
            permuted_x = x.view(-1, self.mnist_dim)[:, self.fixed_permutation]
            return self.model(permuted_x)
        else:
            return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log("loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.default_batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.default_batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.default_batch_size, num_workers=8)
