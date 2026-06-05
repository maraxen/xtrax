"""PreemptionHandler for safe signal-based checkpointing on job preemption."""

import signal
import threading
from collections.abc import Callable


class PreemptionHandler:
    """Handle SIGUSR1/SIGTERM preemption signals for distributed training.

    Only rank 0 executes save_fn. save_fn is called at most once, even if
    multiple signals fire. register() is idempotent.
    """

    def __init__(self, save_fn: Callable[[], None], rank: int = 0) -> None:
        """Initialize PreemptionHandler.

        Args:
            save_fn: Callback to invoke when preempted (called at most once).
            rank: Process rank; only rank 0 calls save_fn (default 0).
        """
        self._save_fn = save_fn
        self._rank = rank
        self._registered = False
        self._preempted = threading.Event()
        self._saved = threading.Event()

    @property
    def preempted(self) -> bool:
        """Whether preemption signal has been received."""
        return self._preempted.is_set()

    def register(self) -> None:
        """Register signal handlers for SIGUSR1 and SIGTERM.

        Idempotent: multiple calls have no additional effect.
        Only rank 0 registers signal handlers.
        """
        if self._registered:
            return
        if self._rank == 0:
            signal.signal(signal.SIGUSR1, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        self._registered = True

    def _handle(self, signum: int, frame) -> None:
        """Signal handler for SIGUSR1 and SIGTERM.

        Sets preempted flag and calls save_fn at most once.
        """
        self._preempted.set()
        if not self._saved.is_set():
            self._saved.set()
            self._save_fn()
