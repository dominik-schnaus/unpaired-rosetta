"""Raw results, stored as one JSON file per run with its configuration, metrics and software versions."""

import dataclasses
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

PACKAGES = ["torch", "numpy", "scipy", "scikit-learn", "pylibmgm"]


def run_id(
    config: object,
) -> str:
    """Stable identifier of a run configuration (a frozen dataclass)."""
    payload = json.dumps({"type": type(config).__name__, **dataclasses.asdict(config)}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def result_path(
    results_dir: Path,
    config: object,
) -> Path:
    return Path(results_dir) / f"{run_id(config)}.json"


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_result(
    results_dir: Path,
    config: object,
    metrics: dict[str, object],
    seconds: float,
) -> Path:
    path = result_path(results_dir, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "config": dataclasses.asdict(config),
        "metrics": metrics,
        "seconds": seconds,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "versions": {package: version(package) for package in PACKAGES},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1, default=str))
    temporary.replace(path)
    return path


def load_results(
    results_dir: Path,
) -> pd.DataFrame:
    """All runs of an experiment as a DataFrame with one column per configuration field and scalar metric."""
    import pandas as pd

    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        record = json.loads(path.read_text())
        scalars = {key: value for key, value in record["metrics"].items() if isinstance(value, (int, float))}
        rows.append({**record["config"], **scalars})
    return pd.DataFrame(rows)
