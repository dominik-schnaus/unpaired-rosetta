"""One-to-one assignments with maximum total score, using SciPy's ``linear_sum_assignment``."""

from concurrent.futures import ThreadPoolExecutor

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def hungarian_matching(
    scores: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rows and columns of the assignment with maximum total score. Matches ``min(n, m)`` pairs."""
    rows, columns = linear_sum_assignment(-scores.double().cpu().numpy())
    return torch.as_tensor(rows), torch.as_tensor(columns)


def batched_hungarian_matching(
    queries: Tensor,
    keys: Tensor,
    block_size: int,
    num_threads: int = 1,
) -> tuple[Tensor, Tensor]:
    """Match block ``i`` of the queries to block ``i`` of the keys by the scores ``queries @ keys.T``.

    Both sets are cut into blocks of ``block_size`` rows, so score matrices stay small. The result holds
    ``min(n, m)`` pairs. The caller must shuffle both sets first.

    With ``num_threads > 1`` the assignments run in parallel (SciPy releases the GIL). The score matrices are still
    computed in order in this thread, so the result does not change and at most ``num_threads`` of them are in memory.
    """
    num_blocks = -(-min(queries.shape[0], keys.shape[0]) // block_size)
    starts = [block * block_size for block in range(num_blocks)]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pending = []
        for start in starts:
            if len(pending) == num_threads:
                pending[-num_threads].result()  # limit the score matrices held in memory
            scores = queries[start : start + block_size] @ keys[start : start + block_size].T
            pending.append(executor.submit(hungarian_matching, scores))
        matchings = [future.result() for future in pending]
    rows = torch.cat([block_rows + start for (block_rows, _), start in zip(matchings, starts)])
    columns = torch.cat([block_columns + start for (_, block_columns), start in zip(matchings, starts)])
    return rows, columns
