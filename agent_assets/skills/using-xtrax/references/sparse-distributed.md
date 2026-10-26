> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Sparse / Distributed / Checkpoint (5% of depth — Pointer Pattern)

#### Sparsification: Structured Pruning

Convert a dense model to sparse (BCOO) format at inference time:

```python
from xtrax.sparse import SparseConfig, SparsePolicy, make_sparse_forward_fn, sparsify_model
import equinox as eqx

policy = SparsePolicy(
    config=SparseConfig(nse_budget=8, update_schedule=lambda step: True),
)

# BEFORE jit: sparsify the model  # verify: src/xtrax/sparse/inference.py:44-55
sparse_model = sparsify_model(model, policy)

# RECOMMENDED: closure keeps the sparse model out of the jit partition.
# fn is (model, inputs) -> outputs; the helper returns (inputs) -> outputs.
forward_fn = make_sparse_forward_fn(lambda model, inputs: model(inputs), sparse_model)
result = jax.jit(forward_fn)(x)

# ALTERNATIVE: Pass to eqx.filter_jit (holds BCOO as static)
@eqx.filter_jit
def inference(x):
    return sparse_model(x)

result = inference(x)
```

Verify: `src/xtrax/sparse/inference.py`

🚫 HALTS: `sparsify_model` **cannot** be called inside `jax.jit`.  
Enforcement: `RuntimeError` from `assert_not_tracing` at `src/xtrax/sparse/inference.py:44-55`  
Reason: BCOO structure is non-static, must be created on host.

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
