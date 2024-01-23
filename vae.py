from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

from mmd import MMD


class BaseVAE(ABC, nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Encoder network forward pass."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor):
        """Decoder network forward pass."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Encoder-Decoder forward pass."""
        pass

    @abstractmethod
    def loss(
        self, input: torch.Tensor, target: torch.Tensor, z: torch.Tensor, beta: float
    ):
        """Reconstruction loss (+ regularization)."""
        pass

    def sample(self, num_samples: int = 99) -> torch.Tensor:
        """Randomly sample from the latent space and produce images."""
        z = torch.randn(num_samples, self.latent_size)
        samples = self.decode(z)
        return samples

    def plot(
        self, num_samples: int = 99, title: str = "Samples from z ~ N(0, 1)"
    ) -> None:
        """Plot images."""
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
        fig.subplots_adjust(hspace=0.05)

        images = self.sample(num_samples=num_samples)

        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                ax.imshow(
                    images[i]
                    .reshape(
                        28,
                        28,
                    )
                    .detach()
                    .numpy(),
                    cmap="gray",
                )
                ax.axis("off")
        fig.suptitle(
            title, fontsize=14, y=0.95
        )  # Adjust the y value to move the title closer
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    def plot_embeddings(self, data) -> None:
        """Visualize the latent space."""
        Zs, labels = [], []
        for batch, (X, y) in enumerate(data):
            X, y = X.to("cpu"), y.to("cpu")
            Z = self.encode(X)
            if isinstance(Z, tuple):
                Z = Z[0]
            Zs.append(Z.detach().numpy())
            labels.append(y.detach().numpy())
        Zs = np.concatenate(Zs, axis=0).squeeze()
        labels = np.concatenate(labels, axis=0)
        cmap = plt.cm.get_cmap("tab10", np.max(labels) + 1)
        plt.figure(figsize=(8, 6))
        plt.scatter(Zs[:, 0], Zs[:, 1], c=labels, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label("Class")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("Encoder Network Embeddings")


class MMDVAE(BaseVAE):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 1024, latent_size: int = 10
    ):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        z: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        n_samples = input.size(0)
        # binary cross entropy because MNIST dataset is roughly multivariate Bernoulli distribution
        reconstruction_loss = F.binary_cross_entropy(input, target, reduction="mean")
        # zero-mean, univariate gaussian as choosen MMD dist
        samples = Variable(
            torch.randn(n_samples, z.size(-1)),
            requires_grad=False,
        ).to(z.device)
        mmd = MMD(samples, z.squeeze())
        return beta * mmd + reconstruction_loss


class VAE(BaseVAE):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 1024, latent_size: int = 10
    ):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.encoder(x)
        mu, log_sigma = torch.split(output, self.latent_size, dim=-1)
        return mu, log_sigma

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma * 0.5)
        z = mu + sigma * torch.randn_like(sigma)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        return self.decode(z), (mu, log_sigma)

    def loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        z: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        mu, log_sigma = z
        # binary cross entropy because MNIST dataset is roughly multivariate Bernoulli distribution
        reconstruction_loss = (
            F.binary_cross_entropy(input, target, reduction="sum") / input.shape[0]
        )
        KLD = (
            -0.5
            * torch.sum(1 + log_sigma - mu**2.0 - torch.exp(log_sigma))
            / input.shape[0]
        )
        return beta * KLD + reconstruction_loss
