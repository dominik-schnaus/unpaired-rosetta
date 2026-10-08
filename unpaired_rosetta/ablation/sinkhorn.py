"""Sinkhorn's algorithm in the log domain."""

import torch
from torch import Tensor


def sinkhorn_projection(
    log_kernel: Tensor,
    row_marginal: Tensor,
    column_marginal: Tensor,
    num_iterations: int,
    tolerance: float = 1e-9,
) -> Tensor:
    """Scale ``exp(log_kernel)`` to ``diag(u) exp(log_kernel) diag(v)`` with the given row and column sums.

    Stops after ``num_iterations`` or once both marginals hold up to ``tolerance`` times their largest entry.
    """
    log_row, log_column = row_marginal.clamp(min=1e-30).log(), column_marginal.clamp(min=1e-30).log()
    f, g = torch.zeros_like(log_row), torch.zeros_like(log_column)
    for _ in range(num_iterations):
        g = log_column - torch.logsumexp(log_kernel + f[:, None], dim=0)
        f = log_row - torch.logsumexp(log_kernel + g[None, :], dim=1)
        plan = torch.exp(log_kernel + f[:, None] + g[None, :])
        column_error = (plan.sum(dim=0) - column_marginal).abs().max()
        row_error = (plan.sum(dim=1) - row_marginal).abs().max()
        if column_error < tolerance * column_marginal.max() and row_error < tolerance * row_marginal.max():
            break
    return torch.exp(log_kernel + f[:, None] + g[None, :])
