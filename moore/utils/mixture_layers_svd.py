# Copyright 2021 The PODNN Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""
SVD version of mixture layers for MOORE MetaWorld experiments.

This file is intentionally added as a new file so the original
Gram-Schmidt implementation remains untouched.
"""

from copy import deepcopy

import torch
import torch.nn as nn


n_models_global = 5
agg_out_dim = 3


class InputLayer(nn.Module):
    """
    InputLayer structures the data in a parallel form ready to be consumed
    by the upcoming parallel layers.
    """

    def __init__(self, n_models):
        super(InputLayer, self).__init__()
        self.n_models = n_models

        global n_models_global
        n_models_global = self.n_models

    def forward(self, x):
        """
        Args:
            x: input tensor with shape [batch, dim]

        Returns:
            x_parallel: [n_models, batch, dim]
        """
        x_parallel = torch.unsqueeze(x, 0)
        x_parallel_next = torch.unsqueeze(x, 0)

        for _ in range(1, self.n_models):
            x_parallel = torch.cat((x_parallel, x_parallel_next), axis=0)

        return x_parallel


class ParallelLayer(nn.Module):
    """
    ParallelLayer creates a parallel layer from the structure of unit_model.
    """

    def __init__(self, unit_model):
        super(ParallelLayer, self).__init__()
        self.n_models = n_models_global
        self.model_layers = []

        for _ in range(self.n_models):
            for j in range(len(unit_model)):
                try:
                    unit_model[j].reset_parameters()
                except Exception:
                    pass
            self.model_layers.append(deepcopy(unit_model))

        self.model_layers = nn.ModuleList(self.model_layers)

    def forward(self, x):
        """
        Args:
            x: [n_models, batch, dim]

        Returns:
            parallel_output: [n_models, batch, dim_out]
        """
        parallel_output = self.model_layers[0](x[0])
        parallel_output = torch.unsqueeze(parallel_output, 0)

        for i in range(1, self.n_models):
            next_layer = self.model_layers[i](x[i])
            next_layer = torch.unsqueeze(next_layer, 0)
            parallel_output = torch.cat((parallel_output, next_layer), 0)

        return parallel_output


def compute_angles(basis):
    """
    Optional debugging helper, same role as in the original file.
    basis expected shape: [batch, n_models, dim]
    """
    cos = torch.abs(torch.clip(basis @ torch.transpose(basis, 1, 2), -1, 1))
    rad = torch.arccos(cos)
    deg = torch.rad2deg(rad)
    off_diagonal_angles = deg[:, ~torch.eye(deg.shape[-1], dtype=bool, device=deg.device)]
    assert torch.all(
        torch.isclose(torch.min(off_diagonal_angles), torch.tensor(90.0, device=deg.device))
    ), torch.min(off_diagonal_angles)


class OrthogonalLayer1D(nn.Module):
    """
    SVD-based orthogonalization layer.

    Input shape:
        [n_models, n_samples, dim]

    Output shape:
        [n_models, n_samples, dim]

    For each sample independently, we take the matrix X with shape
    [n_models, dim] and compute an orthogonalized matrix using SVD:
        X = U S V^T
        X_orth = U V^T
    """

    def __init__(self, eps=1e-8):
        super(OrthogonalLayer1D, self).__init__()
        self.eps = eps

    def forward(self, x):
        # x: [n_models, batch, dim]
        x_bt = torch.transpose(x, 0, 1)  # [batch, n_models, dim]
        basis_all = []

        for i in range(x_bt.shape[0]):
            xi = x_bt[i]  # [n_models, dim]

            # Safe fallback for near-zero tensors
            if torch.linalg.norm(xi) < self.eps:
                basis_all.append(xi)
                continue

            # SVD-based orthogonalization
            # xi = U S Vh  =>  xi_orth = U Vh
            U, _, Vh = torch.linalg.svd(xi, full_matrices=False)
            xi_orth = U @ Vh

            basis_all.append(xi_orth)

        basis = torch.stack(basis_all, dim=0)   # [batch, n_models, dim]
        basis = torch.transpose(basis, 0, 1)    # [n_models, batch, dim]
        return basis