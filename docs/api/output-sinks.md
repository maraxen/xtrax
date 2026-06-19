# Output Sinks

Output sinks are how training results leave the system. xtrax groups two complementary
surfaces under one mental model:

1. **Streaming callbacks** (`xtrax.io`) — bounded async I/O for metrics, logging, and
   other observability without blocking the training loop.
2. **Checkpoint persistence** (`xtrax.checkpoint`) — durable Orbax-backed state for
   resume and fault tolerance.

Callbacks stream live signals during training; checkpoints capture recoverable state
between epochs or steps.

## Streaming Callbacks

The `xtrax.io` module provides async utilities for bounded concurrent execution of
streaming operations.

:::{important}
**Canonical import path** — import from `xtrax.io`, not `xtrax.engine.io`:

```python
from xtrax.io import BoundedCallbackHandler, async_indexed_stream
```

`xtrax.io` is a thin re-export of the implementation in `xtrax.engine.io`.
:::

### Overview

```{autosummary}
xtrax.io.BoundedCallbackHandler
xtrax.io.async_indexed_stream
```

### BoundedCallbackHandler

Manages bounded concurrent execution of async callbacks using an `asyncio.Semaphore`.
Exceptions in submitted coroutines are logged but not propagated, allowing training
loops to continue robustly.

Use this when you need to run I/O-bound operations (logging, metric reporting,
checkpoint triggers) without blocking the training loop, while limiting concurrency
to avoid resource exhaustion.

```python
from xtrax.io import BoundedCallbackHandler
import asyncio

async def main():
    handler = BoundedCallbackHandler(max_concurrent=4)

    async def log_metric(name: str, value: float):
        await asyncio.sleep(0.01)
        print(f"{name}: {value}")

    for i in range(10):
        await handler.submit(log_metric(f"loss_step_{i}", 0.5 - i * 0.01))

    await handler.wait_all()

asyncio.run(main())
```

`Engine` uses `BoundedCallbackHandler` internally for async `on_step_end` callbacks.

### async_indexed_stream

Async iterator that prefetches items from a blocking iterable into a queue, yielding
`(index, item)` tuples. Uses `asyncio.to_thread` so slow blocking I/O does not stall
the event loop.

```python
from xtrax.io import async_indexed_stream
import asyncio

async def main():
    async for index, item in async_indexed_stream(["batch_a", "batch_b"]):
        print(index, item)

asyncio.run(main())
```

### Full API Reference

```{automodule} xtrax.io
:members:
:undoc-members:
:show-inheritance:
```

## Checkpoint Persistence

The `xtrax.checkpoint` module wraps [Orbax CheckpointManager](https://orbax.readthedocs.io/en/latest/)
for saving and restoring training state.

**Canonical import path:**

```python
from xtrax.checkpoint import (
    get_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
)
```

`Engine.fit(..., checkpoint_dir=...)` calls these helpers after each epoch when a
checkpoint directory is provided.

### Overview

```{autosummary}
xtrax.checkpoint.get_checkpoint_manager
xtrax.checkpoint.save_checkpoint
xtrax.checkpoint.load_checkpoint
```

### Checkpoint Round-Trip Example

```python
from pathlib import Path
from xtrax.checkpoint import get_checkpoint_manager, save_checkpoint, load_checkpoint
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    manager = get_checkpoint_manager(directory=tmpdir, max_to_keep=3)

    state_template = {"params": {"w": 0.0}, "step": 0}

    state = {"params": {"w": 1.5}, "step": 100}
    save_checkpoint(manager, state, step=100)

    loaded = load_checkpoint(manager, state_template, step=100)
    assert loaded["step"] == 100
    assert loaded["params"]["w"] == 1.5

    state = {"params": {"w": 2.5}, "step": 200}
    save_checkpoint(manager, state, step=200)

    latest = load_checkpoint(manager, state_template)
    assert latest["step"] == 200
```

In production, pass a `ResumableState` PyTree rather than a plain dict.

### Full API Reference

```{automodule} xtrax.checkpoint
:members:
:undoc-members:
:show-inheritance:
```

### Orbax Integration

These functions use Orbax `CheckpointManager` and `PyTreeCheckpointHandler`. For custom
handlers, multiprocess sync, or metrics:

- [Orbax CheckpointManager API](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint_manager.html)
- [Orbax Handlers](https://orbax.readthedocs.io/en/latest/handlers.html)
