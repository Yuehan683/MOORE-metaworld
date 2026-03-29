import torch
import torch.nn as nn

# 直接复用原始 GS 工程里的并行输入与并行专家层，避免行为漂移
from moore.utils.mixture_layers import InputLayer, ParallelLayer


class OrthogonalLayerHouseholder1D(nn.Module):
    """
    Householder-QR based orthogonalization layer.

    Input:
        x: [n_models, n_samples, dim]

    Output:
        basis: [n_models, n_samples, dim]

    For each sample, vectors along the expert axis are orthonormalized.
    """

    def __init__(self, eps: float = 1e-8, canonical_sign: bool = True):
        super().__init__()
        self.eps = eps
        self.canonical_sign = canonical_sign

    def forward(self, x):
        """
        Args:
            x: tensor with shape [n_models, n_samples, dim]

        Returns:
            tensor with shape [n_models, n_samples, dim]
        """
        # Original semantics:
        #   x.shape = [E, B, D]
        # We need per-sample orthonormalization across expert axis.
        # Convert to [B, D, E], do reduced QR on the last two dims,
        # then transpose back to [E, B, D].
        A = x.permute(1, 2, 0).contiguous()  # [B, D, E]

        # Householder-QR
        Q, R = torch.linalg.qr(A, mode="reduced")  # Q:[B,D,E], R:[B,E,E]

        # Canonical sign stabilization:
        # force diag(R) >= 0 when possible, to reduce sign ambiguity
        if self.canonical_sign:
            diag = torch.diagonal(R, dim1=-2, dim2=-1)  # [B, E]
            sign = torch.sign(diag)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            Q = Q * sign.unsqueeze(1)

        basis = Q.permute(2, 0, 1).contiguous()  # [E, B, D]
        return basis