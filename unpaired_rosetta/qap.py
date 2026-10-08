"""Quadratic assignment with MPOpt (Hutschenreiter et al., 2021) through the ``pylibmgm`` bindings.

The solver minimizes the Koopmans-Beckmann objective

    Tr(cost_x P cost_y P^T) + Tr(linear P^T) = sum_{i,k} cost_x[i, k] cost_y[pi(i), pi(k)] + sum_i linear[i, pi(i)]

over permutations ``pi`` with ``P[i, pi(i)] = 1``.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    import pylibmgm


def seed_c_rand(
    seed: int,
) -> None:
    """Seed the global ``rand()`` of the C library, which QPBO inside MPOpt uses."""
    ctypes.CDLL(None).srand(ctypes.c_uint(seed % 2**32))


def qap_objective(
    cost_x: Tensor,
    cost_y: Tensor,
    permutation: Tensor,
    linear: Tensor | None = None,
) -> float:
    """``sum_{i,k} cost_x[i, k] cost_y[pi(i), pi(k)] + sum_i linear[i, pi(i)]`` for a permutation ``pi``."""
    objective = (cost_x * cost_y[permutation][:, permutation]).sum()
    if linear is not None:
        objective = objective + linear[torch.arange(len(permutation)), permutation].sum()
    return objective.item()


class MPOptQAPSolver:
    """MPOpt with the settings of the paper.

    Runs batches of 100 dual iterations with 10 greedy generations each. Stops when the relative improvement over
    the last ``k = 100`` batches is below ``p = 0.1``, or after 1000 batches.
    """

    batch_size = 100
    greedy_generations = 10
    stopping_p = 0.1
    stopping_k = 100
    max_batches = 1000

    def solve(
        self,
        cost_x: Tensor,
        cost_y: Tensor,
        linear: Tensor | None,
        seed: int,
    ) -> Tensor:
        """Minimizing permutation of the Koopmans-Beckmann objective.

        Args:
            cost_x: Quadratic cost ``[C, C]`` of the first graph.
            cost_y: Quadratic cost ``[C, C]`` of the second graph.
            linear: Linear cost ``[C, C]``, or None.
            seed: Seed of MPOpt and of the C library's ``rand()``.

        Returns:
            The permutation ``[C]``, where ``permutation[i]`` is the partner of ``i``.
        """
        solver = self.build(cost_x, cost_y, linear, seed)
        return self.permutation(solver.run(), cost_x.shape[0])

    def build(
        self,
        cost_x: Tensor,
        cost_y: Tensor,
        linear: Tensor | None,
        seed: int,
    ) -> pylibmgm.QAPSolver:
        """The seeded pylibmgm solver of the problem with the settings of this object, ready to run.

        It must run right after it is built, because the seed of ``rand()`` is set here.
        """
        import pylibmgm

        num_nodes = cost_x.shape[0]
        unary_costs, pairwise_costs = self.graphical_model_costs(cost_x, cost_y, linear)
        model = pylibmgm.GmModel(
            pylibmgm.Graph(0, num_nodes), pylibmgm.Graph(1, num_nodes), num_nodes**2, num_nodes**4 - num_nodes**2
        )
        for i in range(num_nodes):
            for j in range(num_nodes):
                model.add_assignment(i, j, unary_costs[i][j])
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(i + 1, num_nodes):
                    for l in range(num_nodes):  # noqa: E741
                        if j != l:
                            model.add_edge(i, j, k, l, pairwise_costs[i][j][k][l])

        # pylibmgm reads its class-level seed when a solver is built, so it must be set first. Two solves must
        # not run in threads of one process.
        pylibmgm.QAPSolver.libmpopt_seed = seed
        # QPBO shuffles with the C library's global rand(), which libmpopt never seeds. Without this seed, a solve
        # would depend on the rand() calls of earlier solves in the same process.
        seed_c_rand(seed)
        solver = pylibmgm.QAPSolver(model)
        solver.run_settings.batch_size = self.batch_size
        solver.run_settings.greedy_generations = self.greedy_generations
        solver.stopping_criteria.p = self.stopping_p
        solver.stopping_criteria.k = self.stopping_k
        solver.stopping_criteria.max_batches = self.max_batches
        return solver

    @staticmethod
    def permutation(
        solution: pylibmgm.GmSolution,
        num_nodes: int,
    ) -> Tensor:
        """The permutation of a solution. Fails if MPOpt left a node unassigned."""
        permutation = torch.as_tensor(solution.labeling(), dtype=torch.long)
        if not torch.equal(permutation.sort().values, torch.arange(num_nodes)):
            raise RuntimeError(f"MPOpt returned an incomplete assignment: {permutation.tolist()}")
        return permutation

    @staticmethod
    def graphical_model_costs(
        cost_x: Tensor,
        cost_y: Tensor,
        linear: Tensor | None,
    ) -> tuple[list[list[float]], list[list[list[list[float]]]]]:
        """Costs of the assignments ``i -> j`` and of the pairs of assignments ``(i -> j, k -> l)``.

        The pair cost is ``q[i, j, k, l] + q[k, l, i, j]`` with ``q = cost_x[i, k] cost_y[j, l]``. MPOpt needs
        non-positive costs, so both terms are shifted by their maximum plus one. The shift adds the same constant to the
        objective of every permutation.
        """
        cost_x, cost_y = cost_x.double().cpu(), cost_y.double().cpu()
        linear = torch.zeros_like(cost_x) if linear is None else linear.double().cpu()
        quadratic = cost_x[:, None, :, None] * cost_y[None, :, None, :]
        quadratic = (quadratic - (quadratic.max() + 1)).numpy()
        linear = (linear - (linear.max() + 1)).numpy()
        num_nodes = cost_x.shape[0]
        diagonal = np.arange(num_nodes)
        unary_costs = quadratic[diagonal[:, None], diagonal[None, :], diagonal[:, None], diagonal[None, :]] + linear
        pairwise_costs = quadratic + quadratic.transpose(2, 3, 0, 1)
        return unary_costs.tolist(), pairwise_costs.tolist()
