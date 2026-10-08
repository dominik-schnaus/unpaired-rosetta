"""No-op replacements for `dmf.alerts` (progress notifications only)."""

from collections.abc import Callable
from functools import wraps


def alert(
    func: Callable[..., object] | None = None,
    **_kwargs: object,
) -> Callable[..., object]:
    if func is None:
        return lambda f: alert(f)

    @wraps(func)
    def wrapper(
        *args: object,
        **kwargs: object,
    ) -> object:
        return func(*args, **kwargs)

    return wrapper


def send_alert(
    *args: object,
    **kwargs: object,
) -> None:
    return None


def send_message(
    *args: object,
    **kwargs: object,
) -> None:
    return None
