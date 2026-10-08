"""Command line shared by all experiment scripts.

    python -m experiments.<name> run [--shard 0/4] [--workers 8] [--threads 4] [--dry-run]
    python -m experiments.<name> run --slurm --jobs 50
    python -m experiments.<name> plot

``run`` runs every configuration that has no result file yet. ``--shard i/n`` runs every n-th of them, starting at
i, so that n machines can share the runs. ``--slurm --jobs n`` submits n single-node SLURM jobs that share the runs in
the same way. ``plot`` reads the result files and writes the paper's figures and tables as PNG files, each with a CSV
of the plotted numbers. The SLURM partition, a limit on the CPU cores per job and nodes to avoid are set by
environment variables (see ``unpaired_rosetta/settings.py``).
"""

import argparse
import dataclasses
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from unpaired_rosetta import settings
from unpaired_rosetta.embeddings import storage_root
from unpaired_rosetta.results import result_path, save_result


def results_dir(
    experiment: str,
) -> Path:
    """``$UNPAIRED_ROSETTA_ROOT/results/<experiment>``."""
    return storage_root() / settings.results_folder_name() / experiment


def figures_dir(
    experiment: str,
) -> Path:
    """``$UNPAIRED_ROSETTA_ROOT/figures/<experiment>``."""
    path = storage_root() / settings.figures_folder_name() / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_CPUS = 4


@dataclass
class Experiment:
    """A paper experiment: its configurations, how to run one, and how to plot its figures and tables."""

    name: str
    configurations: Callable[[], Sequence]  # frozen dataclasses, one per run
    run: Callable  # (config, num_workers, num_threads) -> metrics dict
    plot: Callable[[], None]
    cpus: int = DEFAULT_CPUS  # SLURM resources per job
    gpus: int = 0
    memory_gb: int = 64
    hours: int = 24

    def selected(
        self,
        only: str | None = None,
    ) -> list:
        """The configurations for which the expression ``only`` holds, e.g. ``"method == 'vec2vec'"``.

        The expression can use the configuration's fields as variables. A field that a configuration does not have is
        None, so one expression works for experiments with several kinds of configurations.
        """
        return [
            config
            for config in self.configurations()
            if only is None or eval(only, {}, defaultdict(lambda: None, dataclasses.asdict(config)))
        ]

    def pending(
        self,
        only: str | None = None,
    ) -> list:
        return [config for config in self.selected(only) if not result_path(results_dir(self.name), config).exists()]


def run_shard(
    experiment: Experiment,
    shard: int,
    num_shards: int,
    num_workers: int,
    num_threads: int,
    only: str | None = None,
) -> None:
    for index, config in enumerate(experiment.selected(only)):  # stable order, so shards never overlap
        if index % num_shards != shard or result_path(results_dir(experiment.name), config).exists():
            continue
        print(f"[{experiment.name}] {config}", flush=True)
        start = time.time()
        try:
            metrics = experiment.run(config, num_workers, num_threads)
        except Exception:
            traceback.print_exc()
            continue
        save_result(results_dir(experiment.name), config, metrics, time.time() - start)


def submit_to_slurm(
    experiment: Experiment,
    num_shards: int,
    num_workers: int,
    num_threads: int,
    exclude: str,
    only: str | None,
    gpus: int,
    cpus: int,
    from_shard: int = 0,
) -> None:
    """Submit one single-node job per shard. Ask for GPUs only if the runs need them."""
    max_cpus = settings.slurm_max_cpus()
    import submitit

    executor = submitit.AutoExecutor(folder=storage_root() / "logs" / experiment.name)
    executor.update_parameters(
        name=experiment.name,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=cpus if max_cpus is None else min(cpus, max_cpus),
        gpus_per_node=gpus,
        mem_gb=experiment.memory_gb,
        timeout_min=experiment.hours * 60,
        slurm_exclude=",".join(filter(None, [settings.slurm_exclude(), exclude])),
    )
    if settings.slurm_partition() is not None:
        executor.update_parameters(slurm_partition=settings.slurm_partition())
    shards = list(range(from_shard, num_shards))
    jobs = executor.map_array(
        run_shard,
        [experiment] * len(shards),
        shards,
        [num_shards] * len(shards),
        [num_workers] * len(shards),
        [num_threads] * len(shards),
        [only] * len(shards),
    )
    print(f"Submitted {len(jobs)} jobs: {jobs[0].job_id.split('_')[0]}")


def main(
    experiment: Experiment,
    arguments: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=experiment.name)
    parser.add_argument("command", choices=["run", "plot"])
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th pending configuration starting at i")
    parser.add_argument("--slurm", action="store_true", help="submit SLURM jobs instead of running here")
    parser.add_argument("--jobs", type=int, default=None, help="with --slurm: number of jobs that share the runs")
    parser.add_argument("--workers", type=int, default=1, help="processes for independent initialization restarts")
    parser.add_argument("--threads", type=int, default=1, help="threads for independent read-out assignments")
    parser.add_argument(
        "--exclude", default="", help="SLURM nodes to avoid, in addition to $UNPAIRED_ROSETTA_SLURM_EXCLUDE"
    )
    parser.add_argument(
        "--only", default=None, help="Python expression over the configuration fields, e.g. \"method == 'ours'\""
    )
    parser.add_argument("--gpus", type=int, default=None, help="GPUs per SLURM job (default: the experiment's)")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs per SLURM job (default: the experiment's)")
    parser.add_argument("--from-shard", type=int, default=0, help="with --slurm: submit only shards from this one on")
    parser.add_argument("--dry-run", action="store_true", help="only list the pending configurations")
    args = parser.parse_args(arguments)

    if args.command == "plot":
        experiment.plot()
        return
    pending = experiment.pending(args.only)
    print(f"{experiment.name}: {len(pending)} of {len(experiment.selected(args.only))} selected runs pending")
    if args.dry_run:
        for config in pending:
            print(config)
        return
    shard, num_shards = (int(part) for part in args.shard.split("/"))
    if args.slurm:
        num_shards = args.jobs or num_shards
        gpus = experiment.gpus if args.gpus is None else args.gpus
        cpus = experiment.cpus if args.cpus is None else args.cpus
        submit_to_slurm(
            experiment, num_shards, args.workers, args.threads, args.exclude, args.only, gpus, cpus, args.from_shard
        )
    else:
        run_shard(experiment, shard, num_shards, args.workers, args.threads, args.only)
