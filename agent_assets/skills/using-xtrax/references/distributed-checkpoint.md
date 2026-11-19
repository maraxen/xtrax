> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Distributed / Checkpoint (5% of depth — Pointer Pattern)

#### Distributed: Multi-Device Training

Initialize distributed context:

```python
import jax

# Start the processes (cluster environments are auto-detected), then build the mesh.
jax.distributed.initialize()
mesh = jax.make_mesh((jax.device_count(),), ("data",))
```

`init_dist`, `is_distributed`, `get_device_mesh`, `get_hardware_mesh_profile`, and
`ShardingPolicy` are deprecated (xtrax #6107): each warns and behaves as before.
`DataModule(distributed=True)` needs `jax.process_count() > 1`.

Verify: `src/xtrax/distributed/init.py`, `src/xtrax/distributed/sharding.py`

#### Checkpoint: Save/Load Training State

Persist training state for resumption:

```python
from xtrax.checkpoint import get_checkpoint_manager, load_checkpoint, save_checkpoint

manager = get_checkpoint_manager("/path/to/ckpt")
save_checkpoint(manager, state, step=int(state.step))
state = load_checkpoint(manager, state_template=state)
```

Verify: `src/xtrax/checkpoint/orbax.py`
