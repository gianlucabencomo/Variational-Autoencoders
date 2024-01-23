import torch
from typing import Callable


def K(xa: torch.Tensor, xb: torch.Tensor):
    # Kernel adapted from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
    xa_size = xa.size(0)
    xb_size = xb.size(0)
    dim = xa.size(1)
    xa = xa.unsqueeze(1)  # (x_size, 1, dim)
    xb = xb.unsqueeze(0)  # (1, y_size, dim)
    tiled_xa = xa.expand(xa_size, xb_size, dim)
    tiled_xb = xb.expand(xa_size, xb_size, dim)
    kernel_input = (tiled_xa - tiled_xb).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def MMD(p: torch.Tensor, q: torch.Tensor, kernel: Callable = K):
    K_pp = kernel(p, p)
    K_qq = kernel(q, q)
    K_pq = kernel(p, q)
    return K_pp.mean() + K_qq.mean() - 2.0 * K_pq.mean()
