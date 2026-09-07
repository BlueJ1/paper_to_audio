"""Per-call diagnostics; context-local so web workers do not share warnings."""
from contextvars import ContextVar
import logging

messages: ContextVar[list[str] | None] = ContextVar("narration_messages", default=None)


def fallback(message: str) -> None:
    logging.getLogger(__name__).warning(message)
    target = messages.get()
    if target is not None:
        target.append(message)
