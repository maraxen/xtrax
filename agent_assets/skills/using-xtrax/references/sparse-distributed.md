> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Sparse / Distributed / Checkpoint (5% of depth — Pointer Pattern)

#### Sparsification: Structured Pruning

Convert a dense model to sparse (BCOO) format at inference time:

```python
from xtrax.sparse import sparsify_model, make_sparse_forward_fn  # verify: src/xtrax/sparse/inference.py
from xtrax.sparse.policy import SparsePolicy
import equinox as eqx

policy = SparsePolicy(target_sparsity=0.9)

# BEFORE jit: sparsify the model  # verify: src/xtrax/sparse/inference.py:44-55
sparse_model = sparsify_model(model, policy)

# RECOMMENDED: Use closure pattern
forward_fn = make_sparse_forward_fn(sparse_model)
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
from xtrax import init_dist, is_distributed, LogicalMesh, with_manual_axes

init_dist(backend="xmap")  # or "pjit"

if is_distributed():
    mesh = LogicalMesh(shape=(2, 4))  # 2×4 device mesh
    with with_manual_axes(mesh):
        # Distributed training code
        pass
```

Verify: `src/xtrax/distributed/` (full reference deferred to source)

#### Checkpoint: Save/Load Training State

Persist training state for resumption:

```python
from xtrax import save_checkpoint, load_checkpoint

# Save
save_checkpoint(state, directory="/path/to/ckpt")

# Load
state = load_checkpoint(directory="/path/to/ckpt")
```

Verify: `src/xtrax/checkpoint/` (see orbax docs for full checkpoint manager API)
