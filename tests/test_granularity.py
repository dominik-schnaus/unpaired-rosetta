"""Tests of the size-balanced k-means and the lifting of its clustering (App. E, Fig. 22)."""

import torch
from torch.nn.functional import normalize

from experiments.granularity import balanced_kmeans, lift, nearest_centroids, seed_centroids


def blobs(
    sizes: list[int],
    dimension: int = 8,
    spread: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-norm points around random unit directions, with ``sizes[c]`` points around direction ``c``."""
    generator = torch.Generator().manual_seed(seed)
    directions = normalize(torch.randn(len(sizes), dimension, generator=generator), dim=1)
    points = torch.cat(
        [
            direction + spread * torch.randn(size, dimension, generator=generator)
            for direction, size in zip(directions, sizes)
        ]
    )
    return normalize(points, dim=1), torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))


def test_seeding_picks_distinct_data_points() -> None:
    data, _ = blobs([200, 200, 200])
    centroids = seed_centroids(data, 3, torch.Generator().manual_seed(0))
    is_row = (centroids[:, None, :] == data[None, :, :]).all(dim=2)
    assert is_row.any(dim=1).all()
    assert torch.unique(is_row.long().argmax(dim=1)).numel() == 3


def test_balanced_kmeans_recovers_separated_clusters_of_equal_size() -> None:
    data, truth = blobs([300, 300, 300, 300])
    centroids = balanced_kmeans(data, 4, torch.Generator().manual_seed(0))
    assert torch.allclose(centroids.norm(dim=1), torch.ones(4))
    labels = nearest_centroids(data, centroids, (centroids**2).sum(dim=1))
    for cluster in range(4):  # every true cluster lands in a single k-means cluster
        assert torch.unique(labels[truth == cluster]).numel() == 1
    assert torch.bincount(labels, minlength=4).tolist() == [300] * 4


def test_balanced_kmeans_is_reproducible() -> None:
    data, _ = blobs([500, 100, 50])
    first = balanced_kmeans(data, 5, torch.Generator().manual_seed(3))
    second = balanced_kmeans(data, 5, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_size_penalty_balances_the_clusters() -> None:
    """On blobs of unequal size, the penalized assignment gives much more balanced clusters than the blobs."""
    data, _ = blobs([700, 200, 100], spread=0.3)
    centroids = balanced_kmeans(data, 4, torch.Generator().manual_seed(0))
    counts = torch.bincount(nearest_centroids(data, centroids, (centroids**2).sum(dim=1)), minlength=4)
    assert counts.min() > 50
    assert counts.max() < 700


def test_lift_gives_every_cluster_the_minimum_size() -> None:
    data, _ = blobs([400, 400, 20])
    centroids = torch.cat(
        [
            normalize(data[:400].mean(0), dim=0)[None],
            normalize(data[400:800].mean(0), dim=0)[None],
            normalize(data[800:].mean(0), dim=0)[None],
        ]
    )
    plain = torch.cdist(data, centroids).argmin(dim=1)
    assert torch.bincount(plain, minlength=3)[2] == 20
    labels = lift(data, centroids, min_size=100)
    counts = torch.bincount(labels, minlength=3)
    assert counts.min() >= 100
    moved = (labels != plain).sum()
    assert moved == 100 - 20 + (counts[2] - 100)  # only rows that join the short cluster leave their nearest center


def test_lift_keeps_the_nearest_center_when_every_cluster_is_large_enough() -> None:
    data, _ = blobs([100, 100, 100])
    centroids = seed_centroids(data, 3, torch.Generator().manual_seed(1))
    plain = ((data**2).sum(dim=1, keepdim=True) + (centroids**2).sum(dim=1) - 2 * data @ centroids.T).argmin(dim=1)
    if torch.bincount(plain, minlength=3).min() >= 10:
        assert torch.equal(lift(data, centroids, min_size=10), plain)
