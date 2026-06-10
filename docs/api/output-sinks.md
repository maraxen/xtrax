# Output Sinks

Output sinks manage how training results and state leave the system. This chapter covers two complementary surfaces:

1. **Streaming Callbacks**: Observability and bounded concurrent operations (e.g., logging, checkpointing events)
2. **Checkpoint Persistence**: Durable state serialization and restoration for training resumption

These two surfaces work together: callbacks stream observability during training, while checkpoints capture state for recovery.

## Streaming Callbacks

The `xtrax.io` module provides async utilities for bounded concurrent execution of streaming operations.

```{important}
**Canonical import path:**
```python
from xtrax.io import BoundedCallbackHandler, async_indexed_stream
```

These are thin re-exports from the canonical implementation in `xtrax.engine.io`. The `xtrax.io` module is the public-facing surface; `xtrax.engine.io` is the implementation. Import from `xtrax.io`.
```

### Overview

```{autosummary}
xtrax.io.BoundedCallbackHandler
xtrax.io.async_indexed_stream
```

### BoundedCallbackHandler

Manages bounded concurrent execution of async callbacks using an `asyncio.Semaphore`. Exceptions in submitted coroutines are logged but not propagated, allowing training loops to continue robustly.

Use this when you need to run I/O-bound operations (logging, metric reporting, checkpoint triggers) without blocking the training loop, while limiting concurrency to avoid resource exhaustion.

```python
from xtrax.io import BoundedCallbackHandler
import asyncio

async def main():
    handler = BoundedCallbackHandler(max_concurrent=4)

    async def log_metric(name: str, value: float):
        # Simulate I/O work (e.g., write to metric store)
        await asyncio.sleep(0.01)
        print(f"{name}: {value}")

    # Submit callbacks - returns immediately
    for i in range(10):
        await handler.submit(log_metric(f"loss_step_{i}", 0.5 - i * 0.01))

    # Wait for all to complete
    await handler.wait_all()

asyncio.run(main())
```

### async_indexed_stream

Async iterator that prefetches items from a blocking iterable into a queue, yielding `(index, item)` tuples. Uses `asyncio.to_thread` to prevent the event loop from being blocked by slow I/O (e.g., file reads, database queries).

Use this when you need to parallelize I/O fetching with CPU-bound computation in an async training loop.

### Full API Reference

```{automodule} xtrax.io
:members:
:undoc-members:
:show-inheritance:
```

## Checkpoint Persistence

The `xtrax.checkpoint` module provides utilities for saving and restoring training state using [Orbax CheckpointManager](https://orbax.readthedocs.io/en/latest/), a composable checkpoint library.

**Canonical import path:**
```python
from xtrax.checkpoint import (
    get_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
)
```

### Overview

```{autosummary}
xtrax.checkpoint.get_checkpoint_manager
xtrax.checkpoint.save_checkpoint
xtrax.checkpoint.load_checkpoint
```

### Checkpoint Round-Trip Example

This example demonstrates the full cycle: creating a manager, saving state, and restoring it.

```python
from pathlib import Path
from xtrax.checkpoint import get_checkpoint_manager, save_checkpoint, load_checkpoint
import tempfile

# Create a temporary directory for checkpoints
with tempfile.TemporaryDirectory() as tmpdir:
    # 1. Create a checkpoint manager
    manager = get_checkpoint_manager(directory=tmpdir, max_to_keep=3)

    # 2. Define a ResumableState (JAX pytree)
    # For this example, use a simple dict; in practice, use your state object
    state_template = {
        "params": {"w": 0.0},
        "step": 0,
    }

    # 3. Save the first checkpoint
    state = {"params": {"w": 1.5}, "step": 100}
    save_checkpoint(manager, state, step=100)

    # 4. Load the checkpoint
    loaded = load_checkpoint(manager, state_template, step=100)
    assert loaded["step"] == 100
    assert loaded["params"]["w"] == 1.5

    # 5. Save another checkpoint
    state = {"params": {"w": 2.5}, "step": 200}
    save_checkpoint(manager, state, step=200)

    # 6. Load the latest checkpoint (no step specified)
    latest = load_checkpoint(manager, state_template)
    assert latest["step"] == 200
```

### Full API Reference

```{automodule} xtrax.checkpoint
:members:
:undoc-members:
:show-inheritance:
```

### Orbax Integration

These functions wrap Orbax's `CheckpointManager` and `PyTreeCheckpointHandler`. For advanced usage (custom handlers, multiprocess synchronization, metrics):

- [Orbax CheckpointManager API](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint_manager.html)
- [Orbax Handlers](https://orbax.readthedocs.io/en/latest/handlers.html)
