"""Import cloned official code in isolation.

Several official repositories have top-level modules with the same name (``utils``). Each import therefore removes
them from ``sys.modules`` before and after, and restores ``sys.path``.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def top_level_names(
    folder: Path,
) -> set[str]:
    return {path.stem for path in folder.glob("*.py")} | {
        path.name for path in folder.iterdir() if (path / "__init__.py").exists()
    }


@contextmanager
def official_code(
    folder: Path,
) -> Iterator[None]:
    """Make the modules of ``folder`` importable for the enclosed block only."""
    names = top_level_names(folder)

    def forget() -> None:
        for name in [name for name in sys.modules if name.split(".")[0] in names]:
            del sys.modules[name]

    forget()
    sys.path.insert(0, str(folder))
    try:
        yield
    finally:
        sys.path.remove(str(folder))
        forget()
