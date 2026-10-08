"""MPOpt, the QAP solver of our initialization, on the class-matching benchmark of Schnaus et al. (2025), for Figure 7.

The solver minimizes ``sum_{i,k} cost1[i, k] cost2[pi(i), pi(k)] + constant = Tr(cost1 P cost2^T P^T) + constant`` over
permutations ``pi`` and returns a ``QAPResult``. All other solvers of Figure 7 are the published runs of Schnaus et al.
(2025), made with their official code (github.com/dominik-schnaus/itsamatch, ``solver_comparison_larger.py``).
"""

import logging
import re
from dataclasses import asdict, dataclass

from torch import Tensor

from unpaired_rosetta.qap import MPOptQAPSolver

# One line per batch of dual iterations: the lower bound, the best cost so far, and the solver time in seconds.
BATCH_LINE = re.compile(r"\bit=\d+ lb=(\S+) ub=(\S+) .*\bt=(\S+)")


@dataclass
class QAPResult:
    permutation: Tensor  # permutation[i] is the partner of i
    cost: float  # objective of the returned permutation
    bound: float | None  # lower bound on the optimum, if the solver gives one
    optimal: bool
    seconds: float
    batches: int = 0  # batches of dual iterations that ran


@dataclass(frozen=True)
class MPOptBudget:
    """Settings of one MPOpt run, named as in ``MPOptQAPSolver``.

    The defaults are the settings of the paper, those of ``MPOptQAPSolver``.
    """

    batch_size: int = MPOptQAPSolver.batch_size
    greedy_generations: int = MPOptQAPSolver.greedy_generations
    max_batches: int = MPOptQAPSolver.max_batches
    stopping_p: float = MPOptQAPSolver.stopping_p
    stopping_k: int = MPOptQAPSolver.stopping_k


def qap_cost(
    cost1: Tensor,
    cost2: Tensor,
    permutation: Tensor,
    constant: float = 0.0,
) -> float:
    return ((cost1 * cost2[permutation][:, permutation]).sum() + constant).item()


class _Lines(logging.Handler):
    def __init__(
        self,
    ) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        self.lines.append(record.getMessage())


def solve_with_bound(
    cost1: Tensor,
    cost2: Tensor,
    constant: float,
    budget: MPOptBudget,
    seed: int,
) -> QAPResult:
    """Run ``MPOptQAPSolver`` with a budget and read its lower bound and time from libmpopt's log.

    pylibmgm passes the log to the ``libmgm`` logger when it runs verbosely. The bound in the log is for the shifted,
    non-positive costs of ``MPOptQAPSolver``. The shift is the same for every permutation, and the returned
    permutation is the one with the last logged best cost ``ub``, so the bound of the original problem is
    ``cost - (ub - lb)`` with the best logged ``lb``. The time counts the dual iterations and the heuristics, not
    building the model.
    """
    solver = MPOptQAPSolver()
    for name, value in asdict(budget).items():
        setattr(solver, name, value)
    mpopt = solver.build(cost1, cost2, None, seed)
    logger = logging.getLogger("libmgm")
    handler = _Lines()
    level, propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # the log has one line per batch
    try:
        solution = mpopt.run(verbose=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
    permutation = MPOptQAPSolver.permutation(solution, cost1.shape[0])
    batches = [match for match in map(BATCH_LINE.search, handler.lines) if match]
    if not batches:
        raise RuntimeError("pylibmgm logged no batch of dual iterations")
    lower = max(float(batch.group(1)) for batch in batches)
    upper, seconds = float(batches[-1].group(2)), float(batches[-1].group(3))
    cost = qap_cost(cost1, cost2, permutation, constant)
    return QAPResult(permutation, cost, cost - (upper - lower), False, seconds, len(batches))
