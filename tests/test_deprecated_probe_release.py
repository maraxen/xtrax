"""#6107 probe release: thin wrappers warn on access and are otherwise unchanged.

For every deprecated name: reaching it raises a `DeprecationWarning` that names the
replacement and is attributed to the CALLER (this file), the object is the unchanged
original, and xtrax's own import raises no warning at all. Nothing was removed.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import xtrax

CODEMOD = Path(__file__).parent.parent / "codemods" / "thin-wrappers-to-jax"
AST_GREP = shutil.which("ast-grep")
needs_ast_grep = pytest.mark.skipif(AST_GREP is None, reason="ast-grep not installed")
# (module the user reaches, name, text the message must contain)
DEPRECATED = [
    (
        "xtrax.safety.manager",
        "SafetyManager",
        "checkify.checkify(fn, errors=checkify.float_checks)",
    ),
    ("xtrax.safety.manager", "with_safety", "err.throw()"),
    ("xtrax.training.step", "SafetyTrainStep", "checkify.checkify(trainer.step"),
    ("xtrax.training", "SafetyTrainStep", "debt #3042"),
    ("xtrax.training.loss", "WeightedLoss", "ComposedLoss"),
    ("xtrax.training", "WeightedLoss", "ComposedLoss"),
    ("xtrax.training.loss", "MultiTaskLoss", "ComposedLoss"),
    ("xtrax.training", "MultiTaskLoss", "ComposedLoss"),
    ("xtrax.training.optim", "partition_labels", "jax.tree.map(lambda leaf"),
    ("xtrax.distributed", "init_dist", "jax.distributed.initialize()"),
    ("xtrax.distributed.init", "init_dist", "jax.distributed.initialize()"),
    ("xtrax.distributed", "is_distributed", "jax.distributed.is_initialized()"),
    ("xtrax.distributed", "get_device_mesh", "jax.make_mesh(shape, axis_names)"),
    ("xtrax.distributed.sharding", "get_device_mesh", "jax.make_mesh(shape, axis_names)"),
    ("xtrax.distributed", "get_hardware_mesh_profile", "jax.make_mesh("),
    ("xtrax.distributed", "ShardingPolicy", "debt #3043"),
    ("xtrax.distributed.sharding", "ShardingPolicy", "debt #3043"),
    ("xtrax.io", "BoundedCallbackHandler", "xtrax.engine.io.BoundedCallbackHandler"),
    ("xtrax.io", "async_indexed_stream", "xtrax.engine.io.async_indexed_stream"),
    # The root re-exports go through the home package's hook.
    ("xtrax", "SafetyTrainStep", "debt #3042"),
    ("xtrax", "WeightedLoss", "ComposedLoss"),
    ("xtrax", "MultiTaskLoss", "ComposedLoss"),
    ("xtrax", "init_dist", "jax.distributed.initialize()"),
    ("xtrax", "is_distributed", "jax.distributed.is_initialized()"),
]


@pytest.mark.parametrize(("module", "name", "replacement"), DEPRECATED)
def test_name_warns_with_replacement_and_names_the_caller(module: str, name: str, replacement: str):
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=r"xtrax #6107") as rec:
        obj = getattr(mod, name)
    assert obj is not None
    assert any(replacement in str(w.message) for w in rec), [str(w.message) for w in rec]
    # Attributed to the caller (this file), or Python's default filter would hide it from
    # the unknown user the probe release exists to find.
    assert {w.filename for w in rec} == {__file__}


@pytest.mark.parametrize(("module", "name", "replacement"), DEPRECATED)
def test_every_import_path_returns_the_same_unchanged_object(
    module: str, name: str, replacement: str
):
    """Class identity and `__name__` are what a consumer dispatching on type sees today."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        obj = getattr(importlib.import_module(module), name)
        home = {
            "SafetyManager": "xtrax.safety.manager",
            "with_safety": "xtrax.safety.manager",
            "SafetyTrainStep": "xtrax.training.step",
            "WeightedLoss": "xtrax.training.loss",
            "MultiTaskLoss": "xtrax.training.loss",
            "partition_labels": "xtrax.training.optim",
            "init_dist": "xtrax.distributed.init",
            "is_distributed": "xtrax.distributed.init",
            "get_device_mesh": "xtrax.distributed.sharding",
            "get_hardware_mesh_profile": "xtrax.distributed.sharding",
            "ShardingPolicy": "xtrax.distributed.sharding",
        }.get(name)
        if home is not None:
            assert obj is getattr(importlib.import_module(home), name)
    assert getattr(obj, "__name__", name) == name


def test_from_import_of_a_deprecated_name_warns():
    with pytest.warns(DeprecationWarning, match="WeightedLoss is deprecated"):
        from xtrax.training import WeightedLoss  # noqa: F401


@pytest.mark.parametrize(
    "module",
    [
        "xtrax.safety.manager",
        "xtrax.training",
        "xtrax.training.step",
        "xtrax.training.loss",
        "xtrax.training.optim",
        "xtrax.distributed",
        "xtrax.distributed.init",
        "xtrax.distributed.sharding",
        "xtrax.io",
    ],
)
def test_unknown_names_still_raise_attribute_error(module: str):
    with pytest.raises(AttributeError, match="NoSuchThing"):
        getattr(importlib.import_module(module), "NoSuchThing")


# ---- behaviour is unchanged ------------------------------------------------------------


def _ignoring_deprecations(module: str, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(importlib.import_module(module), name)


def test_safety_manager_and_with_safety_behave_as_before():
    manager_cls = _ignoring_deprecations("xtrax.safety.manager", "SafetyManager")
    with_safety = _ignoring_deprecations("xtrax.safety.manager", "with_safety")

    def double(x):
        return x * 2.0

    off = manager_cls(enabled=False, check_nans=False, check_infs=False)
    assert off.wrap(double) is double
    on = manager_cls(enabled=True, check_nans=True, check_infs=True)
    assert float(with_safety(double, on)(jnp.asarray(3.0))) == 6.0
    with pytest.raises(Exception, match="division by zero"):
        on.wrap(lambda x: x / 0.0)(jnp.asarray(1.0))


def test_create_train_step_warns_only_on_safety_true():
    import optax

    from xtrax.training import create_train_step

    def loss(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        plain = create_train_step(loss, optax.sgd(0.1))
    assert [w for w in rec if issubclass(w.category, DeprecationWarning)] == []
    assert type(plain).__name__ == "Trainer"

    with pytest.warns(DeprecationWarning, match=r"create_train_step\(safety=True\)") as rec:
        checked = create_train_step(loss, optax.sgd(0.1), safety=True)
    assert {w.filename for w in rec} == {__file__}
    assert "checkify.checkify(trainer.step" in str(rec[0].message)
    assert type(checked).__name__ == "SafetyTrainStep"
    assert type(checked.trainer).__name__ == "Trainer"
    assert type(checked.safety_manager).__name__ == "SafetyManager"


def test_weighted_and_multitask_loss_behave_as_before():
    weighted_cls = _ignoring_deprecations("xtrax.training.loss", "WeightedLoss")
    multi_cls = _ignoring_deprecations("xtrax.training.loss", "MultiTaskLoss")

    def mse(p, t):
        return jnp.mean((p - t) ** 2)

    p, t = jnp.asarray([1.0, 3.0]), jnp.asarray([0.0, 1.0])
    weighted = weighted_cls(mse, weight=2.0)
    assert float(weighted(p, t)) == pytest.approx(2.0 * 2.5)
    multi = multi_cls(losses=(weighted, weighted_cls(mse, weight=1.0)))
    assert float(multi((p, p), (t, t))) == pytest.approx(3.0 * 2.5)


def test_partition_labels_behaves_as_before():
    partition_labels = _ignoring_deprecations("xtrax.training.optim", "partition_labels")
    model = {"w": jnp.ones(2), "b": jnp.zeros(2)}
    labels = partition_labels(model, lambda leaf: leaf.sum() == 0.0)
    assert labels == {"w": "train", "b": "frozen"}
    assert labels == jax.tree.map(lambda leaf: "frozen" if leaf.sum() == 0.0 else "train", model)


def test_distributed_helpers_behave_as_before():
    init_mod = importlib.import_module("xtrax.distributed.init")
    init_dist = _ignoring_deprecations("xtrax.distributed.init", "init_dist")
    is_distributed = _ignoring_deprecations("xtrax.distributed.init", "is_distributed")
    get_device_mesh = _ignoring_deprecations("xtrax.distributed.sharding", "get_device_mesh")
    profile = _ignoring_deprecations("xtrax.distributed.sharding", "get_hardware_mesh_profile")
    policy_cls = _ignoring_deprecations("xtrax.distributed.sharding", "ShardingPolicy")

    init_mod._reset_for_testing()
    try:
        assert is_distributed() is False
        init_dist(coordinator_address="localhost:1234", num_processes=1, process_id=0)
        assert is_distributed() is True
    finally:
        init_mod._reset_for_testing()

    n = len(jax.devices())
    mesh = get_device_mesh((n,), ("x",))
    assert mesh.axis_names == ("x",)
    assert mesh.devices.shape == (n,)
    assert set(profile()) == {
        "device_type",
        "num_devices",
        "recommended_shape",
        "recommended_axis_names",
    }
    spec = jax.sharding.PartitionSpec("x")
    assert policy_cls(rules=(("w", spec),)).get_partition_spec("layer/w") == spec


def test_xtrax_io_returns_the_engine_io_objects():
    engine_io = importlib.import_module("xtrax.engine.io")
    for name in ("BoundedCallbackHandler", "async_indexed_stream"):
        assert _ignoring_deprecations("xtrax.io", name) is getattr(engine_io, name)
    assert importlib.import_module("xtrax.io").callbacks is importlib.import_module(
        "xtrax.io.callbacks"
    )


# ---- controls: xtrax's own code and the replacements do not warn -----------------------


def test_replacements_do_not_warn():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        from xtrax import ComposedLoss, Trainer, create_train_step  # noqa: F401
        from xtrax.engine.io import BoundedCallbackHandler  # noqa: F401
    assert [w for w in rec if issubclass(w.category, DeprecationWarning)] == []


def test_xtrax_own_import_raises_no_deprecation_warning():
    """`python -W error::DeprecationWarning -c "import xtrax"` (and every other module xtrax
    imports for itself) must stay silent: no xtrax code may reach a deprecated name."""
    code = (
        "import xtrax, xtrax.training, xtrax.engine, xtrax.data, xtrax.safety, "
        "xtrax.safety.manager, xtrax.loop.checkified_execution, xtrax.tiling, "
        "xtrax.distributed, xtrax.io, xtrax.io.callbacks"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_every_non_deprecated_public_name_resolves_without_a_warning():
    deprecated = {"SafetyTrainStep", "WeightedLoss", "MultiTaskLoss", "init_dist", "is_distributed"}
    names = [n for n in xtrax.__all__ if n not in deprecated]
    code = f"import xtrax\nfor n in {names!r}:\n    getattr(xtrax, n)\n"
    result = subprocess.run(
        [sys.executable, "-I", "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_control_the_error_filter_does_catch_a_deprecated_name():
    """Negative control for the two subprocess tests above: the same filter turns a
    deprecated access into a failure, so their silence is evidence, not a dead filter."""
    code = "import xtrax; xtrax.WeightedLoss"
    result = subprocess.run(
        [sys.executable, "-I", "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "DeprecationWarning" in result.stderr


def test_checkified_execution_gate_no_longer_uses_safety_manager(tmp_path):
    """The loop gate builds checkify itself, so running it raises no #6107 warning."""
    from xtrax.loop.checkified_execution import (
        CheckifiedExecutionError,
        assert_checkified_execution,
    )

    candidate = tmp_path / "cand.py"
    candidate.write_text(
        "import jax.numpy as jnp\n"
        "def ok(x):\n    return x * 2.0\n"
        "def bad(x):\n    return x / jnp.zeros_like(x)\n"
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = assert_checkified_execution(candidate, "ok", concrete_inputs=[jnp.asarray(2.0)])
        with pytest.raises(CheckifiedExecutionError):
            assert_checkified_execution(candidate, "bad", concrete_inputs=[jnp.asarray(1.0)])
    assert float(out) == 4.0
    assert [w for w in rec if "#6107" in str(w.message)] == []


@needs_ast_grep
def test_codemod_rewrites_the_fixture_exactly_and_is_idempotent(tmp_path: Path):
    target = tmp_path / "fixture.py"
    before = (CODEMOD / "fixture_before.py").read_text()
    target.write_text(before)
    cmd = [AST_GREP, "scan", "--rule", str(CODEMOD / "rules.yml"), "--update-all", str(target)]
    subprocess.run(cmd, check=True, capture_output=True)
    after = target.read_text()
    assert after != before, "control: the rules must change the unmigrated fixture"
    assert after == (CODEMOD / "fixture_after.py").read_text()
    subprocess.run(cmd, check=True, capture_output=True)
    assert target.read_text() == after, "a second run must change nothing"


@needs_ast_grep
def test_codemod_report_only_rules_flag_deprecated_uses_and_spare_safety_false(tmp_path: Path):
    target = tmp_path / "uses.py"
    target.write_text(
        "from xtrax.training import WeightedLoss, create_train_step\n"
        "a = create_train_step(model, opt, safety=True)\n"
        "b = create_train_step(model, opt, safety=False)\n"
    )
    cmd = [AST_GREP, "scan", "--rule", str(CODEMOD / "rules.yml"), "--json", str(target)]
    hits = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    by_rule = {h["ruleId"]: h["text"] for h in hits}
    assert set(by_rule) == {"deprecated-class-import", "create-train-step-safety-true"}
    assert "safety=True" in by_rule["create-train-step-safety-true"]
    assert target.read_text().count("safety=False") == 1  # report rules never rewrite
