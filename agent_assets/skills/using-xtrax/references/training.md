> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Training Layer (25% of depth — ResumableState, Trainer, SafetyTrainStep, Engine, Callbacks, Optax)

#### ResumableState: Training State

Immutable training state that can be saved and restored:

```python
from xtrax.training.types import ResumableState  # verify: src/xtrax/training/types.py

state = ResumableState(
    step=jnp.int32(0),         # jnp.int32 scalar — dynamic leaf; plain 0 (Python int) also accepted
    key=jax.random.key(0),     # PRNG key
    model=my_model,            # eqx.Module (trainable parameters)
    opt_state=None,            # Optimizer state (optax format)
)
```

Verify: `src/xtrax/training/types.py`

**Mutation pattern** (using `eqx.tree_at`):

```python
import equinox as eqx

# Update model and step in state
new_state = eqx.tree_at(
    lambda s: (s.model, s.step),
    state,
    (new_model, state.step + 1),
)
```

Verify: `src/xtrax/training/trainer.py:67-70`

#### Trainer: Single-Model Training Step

Execute one supervised training step:

```python
from xtrax.training.types import LossFunction  # verify: src/xtrax/training/types.py
from xtrax.training.trainer import Trainer  # verify: src/xtrax/training/trainer.py:12-74
import optax

loss_fn: LossFunction = lambda pred, target: jnp.mean((pred - target) ** 2)
optimizer = optax.adam(learning_rate=1e-4)

trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)

# Execute step
new_state, metrics = trainer.step(state, batch)  # verify: src/xtrax/training/trainer.py:31-74
# metrics = {"loss": scalar}
# new_state.step incremented by 1
```

Verify: `src/xtrax/training/trainer.py:12-74`

**Invariant**: `Trainer.step` is `@eqx.filter_jit` decorated (trace only JAX arrays).

#### SafetyTrainStep: Gradient Safety

Numerical safety wrappers for gradient computation:

```python
from xtrax.safety.manager import SafetyManager
from xtrax.training.step import SafetyTrainStep

# NaN/Inf checks. Gradient clipping is make_optimizer(base, clip_norm=).
safety = SafetyTrainStep(
    trainer=trainer,
    safety_manager=SafetyManager(enabled=True, check_nans=True, check_infs=True),
)
```

Verify: `src/xtrax/training/step.py`, `src/xtrax/safety/manager.py`

#### Engine: Async Training Loop

High-level training orchestration with callbacks:

```python
from xtrax.engine.engine import Engine

engine = Engine(
    trainer=trainer,
    callbacks=(callback1, callback2),
)

# data.train_iter() yields batches. Both calls return the final ResumableState.
final_state = engine.fit_sync(state, data, num_epochs=10)


async def train():
    return await engine.fit(state, data, num_epochs=10)
```

Verify: `src/xtrax/engine/engine.py`

⚠ NOTE: `Engine.fit` is async. `fit_sync` is the blocking wrapper. Both return the final `ResumableState`. `data` is a `fit` argument (`DataModule` or any object with `train_iter`).

#### Callback Protocol

Extend training with custom hooks:

```python
from xtrax.engine.engine import Engine
from xtrax.training.types import Callback

class LoggingCallback(Callback):
    """Log metrics every N steps."""
    
    def on_step_end(self, state, metrics):
        if state.step % 100 == 0:
            print(f"Step {state.step}: {metrics}")

    def on_epoch_end(self, state, epoch: int):
        print(f"Epoch {epoch} end")

trainer = Trainer(...)
engine = Engine(trainer=trainer, callbacks=(LoggingCallback(),))
```

Verify: `src/xtrax/training/types.py`

**Callback hooks** (7 total, verify: `src/xtrax/training/types.py:32-40`):
- `on_train_start(state)` — Before training begins
- `on_train_end(state)` — After all training
- `on_resume(state)` — When resuming from a checkpoint
- `on_epoch_start(state, epoch: int)` — Before epoch (note `epoch` arg)
- `on_epoch_end(state, epoch: int)` — After epoch (note `epoch` arg)
- `on_step_start(state)` — Before step
- `on_step_end(state, metrics)` — After step, receives metrics dict

⚠ NOTE: Callback hooks run **Python-side, outside JAX traces**. Mutating state in callbacks has no effect on training.

#### Optax Integration

Create learning rate schedules and optimizer chains:

```python
from xtrax.training import make_optimizer, adamw_with_schedule
import optax

# Simple Adam
opt = optax.adam(learning_rate=1e-4)

# Adam with learning rate schedule
schedule = optax.exponential_decay(
    init_value=1e-4,
    transition_steps=1000,
    decay_rate=0.96,
)
opt_with_schedule = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate=schedule),
)

# make_optimizer wraps a base GradientTransformation with optional clipping.
base = optax.adamw(learning_rate=1e-4)
opt = make_optimizer(base, clip_norm=1.0)
opt = adamw_with_schedule(peak_lr=1e-4, warmup_steps=1000, total_steps=10000)
```

Verify: `src/xtrax/training/optim.py` (definitions); re-exported at `src/xtrax/training/__init__.py:4`
