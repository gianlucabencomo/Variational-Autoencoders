import typer
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from vae import VAE, MMDVAE
from helper import set_random_seeds, load_data


def train(
    data: DataLoader,
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim,
    epochs: int,
    device: str,
    verbose: bool = False,
) -> None:
    """VAE training loop.

    Trains variational autoencoder (VAE) of two different flavors: MMD-VAE
    and the standard VAE. Data that is currently accepted are 28 x 28
    grayscale images.

    Args:
        data: torch.utils.data.DataLoader
            Dataloader to iterate through for training.

        model: torch.nn.Module
            Model to use for forward passes.

        criterion: Callable
            Loss function to backpropagate through.

        optimizer: torch.optim
            Optimizer to use for updating weights (recommended: Adam).

        device: str
            Device to train on.

        verbose: bool
            Print out progress.

    Returns:
        None. Model parameters are updated without return.

    Raises:
        None.
    """
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch, (X, _) in enumerate(data):
            X = X.to(device)

            X_, Z = model(X)
            loss = criterion(X_, X, Z)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        if epoch % 1 == 0 and verbose:
            print(f"Epoch {epoch + 1} / {epochs} : loss =  {np.mean(losses):>7f}")


def main(
    seed: int = 0,  # random seed
    epochs: int = 30,  # number of training epochs
    batch_size: int = 200,  # batch size
    latent_size: int = 2,  # dimension of z at the bottleneck
    alpha: float = 1e-3,  # learning rate
    beta: float = 1.0,  # scaling param for divergence
    mode: str = "vae",  # "vae" or "mmd"
    verbose: bool = False,  # print feedback
    plot: bool = False,  # make plots at the end
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if verbose:
        print(f"Using {device} device")

    # set up random seeds
    set_random_seeds(seed)

    # load data and setup dataloaders
    trainset, testset = load_data(fashion=False)
    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    # model
    if mode == "vae":
        model = VAE(latent_size=latent_size).to(device)
    elif mode == "mmd":
        model = MMDVAE(latent_size=latent_size).to(device)
    else:
        raise NotImplementedError(
            "Model specified has not been implemented. Choices = mmd or vae."
        )

    # criterion (loss) + optimizer
    criterion = partial(model.loss, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # run training loop
    train(trainloader, model, criterion, optimizer, epochs, device, verbose)

    # prepare for eval
    model = model.to("cpu")

    if plot:
        model.eval()
        model.plot()
        model.plot_embeddings(testloader)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
