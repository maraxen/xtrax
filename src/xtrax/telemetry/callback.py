"""Telemetry callback: the seam between ``Engine``'s hooks and the run ledger.

``xtrax.training.types.Callback`` already defines seven hooks, and ``Engine``
already fires them for training (``callbacks``) and validation (``eval`` ->
``validation_callbacks``). Both call sites in the CLI hard-coded ``callbacks=()``
-- the seam existed and was simply unused. This class fills it rather than
introducing a parallel mechanism, so one object instruments both the training and
the inference path.

Structural note on IR capture. The compile boundary is the right place to
observe, but the callback protocol never sees a batch: ``on_step_start`` takes
only the state. Rather than widen the protocol, the caller supplies a
zero-argument thunk closing over whatever it needs (typically ``trainer.step``
and the first real batch), and this class invokes it exactly once, on the first
step. That first step *is* the compile, so the capture lands at the boundary with
the true shape signature of the workload -- and the once-only guard is what keeps
one blob per signature instead of one per step.
"""

import time
from typing import Any

from xtrax.telemetry.ir import capture_ir, degraded_reason
from xtrax.telemetry.ledger import RunLedger
from xtrax.telemetry.record import STATUS_DEGRADED, IRRef
from xtrax.telemetry.store import BlobStore


class TelemetryCallback:
    """Records IR and run progress into an open :class:`RunLedger`.

    Satisfies ``xtrax.training.types.Callback`` structurally (it is a Protocol,
    so no inheritance is required). Every hook is deliberately cheap: the
    expensive work -- git shellouts, pinning -- already happened once in
    ``RunLedger.open``, and must never recur per step.
    """

    def __init__(
        self,
        ledger: RunLedger,
        *,
        ir_capture: "Any | None" = None,
    ) -> None:
        self.ledger = ledger
        self._ir_capture = ir_capture
        self._captured = False
        self.epochs = 0
        self.steps = 0
        self.started_at: float | None = None

    # -- IR ------------------------------------------------------------------

    def capture_ir_once(self) -> None:
        """Invoke the IR thunk on first call; a no-op on every later call.

        Guarded rather than idempotent-by-luck: without the flag this would fire
        every step, and since each capture would hash identical text the blob
        store would dedup it -- hiding an O(steps) tracing cost behind a
        correct-looking store. The guard makes the cost visible in the design
        instead of invisible in a profile.
        """
        if self._captured or self._ir_capture is None:
            return
        self._record(self._ir_capture())

    def capture_ir_for(self, fn: Any, *args: Any) -> None:  # noqa: ANN401
        """Capture ``fn``'s IR at ``args`` on first call; a no-op thereafter.

        The direct form used by ``Engine``, which -- unlike a callback -- has the
        step function and a real batch in hand. Same once-only guard as
        :meth:`capture_ir_once`, and for the same reason: the first call *is* the
        compile, so capturing there lands at the boundary with the true shape
        signature, and every later call would re-trace for a blob that already
        exists.
        """
        if self._captured:
            return
        if self.ledger.opted_out:
            self._captured = True
            return
        self._record(capture_ir(fn, *args, store=BlobStore(self.ledger.blob_root)))

    def _record(self, refs: "tuple[IRRef, ...]") -> None:
        self._captured = True
        if self.ledger.opted_out:
            return
        self.ledger.record_ir(refs)
        reason = degraded_reason(refs)
        if reason is not None:
            self.ledger.set_status(STATUS_DEGRADED, f"IR capture incomplete -- {reason}")

    # -- Callback protocol ---------------------------------------------------

    def on_train_start(self, state: Any) -> None:  # noqa: ANN401, ARG002
        self.started_at = time.monotonic()

    def on_train_end(self, state: Any) -> None:  # noqa: ANN401, ARG002
        """Deliberately does not close the ledger.

        Closing belongs to whoever opened it -- normally the ``with`` block in
        ``Engine`` -- so that a run which raises still writes a row. A callback
        that closed here would miss exactly the failures worth recording.
        """

    def on_resume(self, state: Any) -> None:  # noqa: ANN401, ARG002
        return

    def on_epoch_start(self, state: Any, epoch: int) -> None:  # noqa: ANN401, ARG002
        return

    def on_epoch_end(self, state: Any, epoch: int) -> None:  # noqa: ANN401, ARG002
        self.epochs = max(self.epochs, epoch + 1)

    def on_step_start(self, state: Any) -> None:  # noqa: ANN401, ARG002
        self.capture_ir_once()

    def on_step_end(self, state: Any, metrics: "dict[str, Any]") -> None:  # noqa: ANN401, ARG002
        self.steps += 1

    @property
    def elapsed_seconds(self) -> "float | None":
        if self.started_at is None:
            return None
        return time.monotonic() - self.started_at
