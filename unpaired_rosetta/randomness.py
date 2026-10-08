"""Random generators of a run, all seeded with the run seed.

``torch`` draws permutations and QAP solver seeds, ``numpy`` draws k-means seeds. Both are seeded with the run seed,
which reproduces the paper's numbers.
Baselines built on external code use the global generators, which ``global_generators`` sets to the run's streams.
"""

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor


class Randomness:
    """Seeded torch, numpy and python generators of one run, plus the CUDA state used by external code."""

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = int(seed)
        self.torch = torch.Generator().manual_seed(self.seed)
        self.numpy = np.random.RandomState(self.seed)
        self.python = random.Random(self.seed)
        self.cuda_state: list[Tensor] | None = None  # seeded with the run seed on first use

    def permutation(
        self,
        num_samples: int,
    ) -> Tensor:
        return torch.randperm(num_samples, generator=self.torch)

    def subset(
        self,
        num_samples: int,
        size: int | None,
    ) -> Tensor:
        """A random subset of ``range(num_samples)`` in random order (all indices if ``size`` is None)."""
        return self.permutation(num_samples)[:size]

    def solver_seed(
        self,
    ) -> int:
        return int(torch.randint(0, 2**32 - 1, (1,), generator=self.torch).item())

    def kmeans_seed(
        self,
    ) -> int:
        return int(self.numpy.randint(0, 2**31 - 1))

    @contextmanager
    def global_generators(
        self,
    ) -> Iterator[None]:
        """Use the streams of this run in the global torch, numpy and python generators, for external code.

        On exit the run keeps the advanced streams and the caller's global states are restored.
        """
        saved = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.set_rng_state(self.torch.get_state())
        np.random.set_state(self.numpy.get_state())
        random.setstate(self.python.getstate())
        if torch.cuda.is_available():
            if self.cuda_state is None:
                torch.cuda.manual_seed_all(self.seed)
            else:
                torch.cuda.set_rng_state_all(self.cuda_state)
        try:
            yield
        finally:
            self.torch.set_state(torch.get_rng_state())
            self.numpy.set_state(np.random.get_state())
            self.python.setstate(random.getstate())
            torch.set_rng_state(saved[0])
            np.random.set_state(saved[1])
            random.setstate(saved[2])
            if saved_cuda is not None:
                self.cuda_state = torch.cuda.get_rng_state_all()
                torch.cuda.set_rng_state_all(saved_cuda)


def derive_seeds(
    seed: int = 42,
    num_seeds: int = 5,
) -> list[int]:
    """The seeds of all experiments.

    ``derive_seeds(42, 5) == [734796314, 576165995, 2197670066, 839703249, 2584932063]``.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**32 - 1, (num_seeds,), generator=generator).tolist()


SEEDS = derive_seeds(42, 5)
