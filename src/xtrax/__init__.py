__version__ = "0.4.0a1"

__all__ = [
    # Core training
    "Trainer",
    "SafetyTrainStep",
    "create_train_step",
    "LossFunction",
    "Callback",
    "ResumableState",
    "WeightedLoss",
    "MultiTaskLoss",
    "make_optimizer",
    "adamw_with_schedule",
    # Engine and IO
    "Engine",
    "BoundedCallbackHandler",
    "save_checkpoint",
    "load_checkpoint",
    # Data
    "DataModule",
    "create_distributed_pipeline",
    # Tiling
    "AxisSpec",
    "AxisDecision",
    "BatchPlan",
    "BatchPlanner",
    "Vmap",
    "SafeMap",
    "DedupGather",
    "Bucket",
    "select_bucket",
    "bucketize",
    # Sparse
    "SparseConfig",
    "SparsePolicy",
    "SparseMaskManager",
    "sparsify_model",
    "make_sparse_forward_fn",
    "sparse_filter_jit",
    # Distributed
    "init_dist",
    "is_distributed",
    # Transforms
    "safe_map",
    "safe_scan",
    # Safety
    "safe_norm",
    "safe_reciprocal",
    # Stages
    "TransformFn",
    "RollingFn",
]

_LAZY = {
    # Training subpackage
    "Trainer": "xtrax.training",
    "SafetyTrainStep": "xtrax.training",
    "create_train_step": "xtrax.training",
    "LossFunction": "xtrax.training",
    "Callback": "xtrax.training",
    "ResumableState": "xtrax.training",
    "WeightedLoss": "xtrax.training",
    "MultiTaskLoss": "xtrax.training",
    "make_optimizer": "xtrax.training",
    "adamw_with_schedule": "xtrax.training",
    # Engine subpackage
    "Engine": "xtrax.engine",
    "BoundedCallbackHandler": "xtrax.engine",
    # Checkpoint subpackage
    "save_checkpoint": "xtrax.checkpoint",
    "load_checkpoint": "xtrax.checkpoint",
    # Data subpackage
    "DataModule": "xtrax.data",
    "create_distributed_pipeline": "xtrax.data",
    # Tiling subpackage
    "AxisSpec": "xtrax.tiling",
    "AxisDecision": "xtrax.tiling",
    "BatchPlan": "xtrax.tiling",
    "BatchPlanner": "xtrax.tiling",
    "Vmap": "xtrax.tiling",
    "SafeMap": "xtrax.tiling",
    "DedupGather": "xtrax.tiling.strategy",
    "Bucket": "xtrax.tiling",
    "select_bucket": "xtrax.tiling",
    "bucketize": "xtrax.tiling",
    # Sparse subpackage
    "SparseConfig": "xtrax.sparse",
    "SparsePolicy": "xtrax.sparse",
    "SparseMaskManager": "xtrax.sparse",
    "sparsify_model": "xtrax.sparse",
    "make_sparse_forward_fn": "xtrax.sparse",
    "sparse_filter_jit": "xtrax.sparse",
    # Distributed subpackage
    "init_dist": "xtrax.distributed",
    "is_distributed": "xtrax.distributed",
    # Transforms subpackage
    "safe_map": "xtrax.transforms",
    "safe_scan": "xtrax.transforms",
    # Safety subpackage
    "safe_norm": "xtrax.safety",
    "safe_reciprocal": "xtrax.safety",
    # Stages subpackage
    "TransformFn": "xtrax.stages",
    "RollingFn": "xtrax.stages",
}


def __getattr__(name):
    """Lazy import on attribute access."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'xtrax' has no attribute {name!r}")


def __dir__():
    """Return sorted list of public attributes."""
    return sorted(__all__)
