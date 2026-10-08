"""No-op replacement for `dmf.env`."""

import os


def getenv(
    key: str,
    default: str | None = None,
) -> str | None:
    return os.environ.get(key, default)
