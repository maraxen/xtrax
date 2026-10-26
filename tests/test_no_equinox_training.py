"""No-equinox gate for the training path (equinox decoupling, step 3).

Spec ``.praxia/docs/specs/261023_equinox-decoupling.md`` section 6. A subprocess puts a
meta-path finder in front of ``sys.meta_path`` that raises on ``import equinox``, imports
every ``xtrax`` module, and trains a dict model through ``Trainer``, ``SafetyTrainStep``,
``accumulate_grads``, ``Engine`` and an orbax checkpoint. ``equinox`` stays a declared
dependency until step 5; this test is about whether xtrax's code reaches for it.

Modules that still import equinox are listed in ``_PENDING`` (``RunSpec`` moves in step 4).
The list may only shrink: a module listed there that no longer touches equinox fails
``test_pending_list_is_not_stale``, so step 4 has to delete its entries.
"""

import ast
import json
import pathlib
import subprocess
import sys

import pytest

import xtrax

_SRC = pathlib.Path(xtrax.__file__).parent

# ``xtrax.run.spec`` and ``xtrax.run.resolver`` import equinox until step 4, and importing any
# ``xtrax.run.*`` module runs the package ``__init__`` that imports them. The cli verbs and loop
# gates import ``xtrax.run`` in turn. None of these is on the training path.
_PENDING = {
    "xtrax.cli.entrypoint",
    "xtrax.cli.registry",
    "xtrax.cli.run",
    "xtrax.cli.run_verb",
    "xtrax.cli.sweep_verb",
    "xtrax.loop.campaign_approval_gate",
    "xtrax.loop.evaluator_change_gate",
    "xtrax.loop.evaluator_completeness",
    "xtrax.loop.info_barrier_lint",
    "xtrax.loop.metrics_provenance",
    "xtrax.loop.promotion_gate",
    "xtrax.loop.scope_expansion_gate",
    "xtrax.run",
    "xtrax.run.freshness",
    "xtrax.run.memory_sink",
    "xtrax.run.repro_floor",
    "xtrax.run.resolver",
    "xtrax.run.seed_emission",
    "xtrax.run.sink",
    "xtrax.run.spec",
    "xtrax.run.zarr_commit",
    "xtrax.run.zarr_sink",
}

_GATE = """
import asyncio
import importlib
import importlib.abc
import importlib.machinery
import json
import pathlib
import sys
import os
import tempfile
import warnings

ATTEMPTS = []


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "equinox" or name.startswith("equinox."):
            ATTEMPTS.append(name)
            raise ImportError("equinox import blocked")
        return None


class Inject(importlib.abc.MetaPathFinder):
    # Appends `import equinox` to the source of one real module, bypassing bytecode caches.
    def __init__(self, target):
        self.target = target

    def find_spec(self, name, path=None, target=None):
        if name != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)

        class Loader(importlib.machinery.SourceFileLoader):
            def get_code(self, fullname):
                source = self.get_data(self.path) + b"\\nimport equinox\\n"
                return self.source_to_code(source, self.path)

        spec.loader = Loader(spec.loader.name, spec.loader.path)
        return spec


sys.meta_path.insert(0, Block())
INJECT = INJECT_TARGET
if INJECT:
    sys.meta_path.insert(1, Inject(INJECT))

import xtrax

root = pathlib.Path(xtrax.__file__).parent
names = []
for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root).with_suffix("")
    parts = ("xtrax", *rel.parts)
    names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))

touched = set()
for name in names:
    ATTEMPTS.clear()
    try:
        importlib.import_module(name)
    except Exception as exc:
        if "equinox import blocked" in str(exc):
            touched.add(name)
        continue
    if ATTEMPTS:  # swallowed by a try/except around the import
        touched.add(name)

def exercise():
    # Tiny train + resume on a dict model (no equinox anywhere).
    import jax
    import jax.numpy as jnp
    import optax

    from xtrax.checkpoint.orbax import get_checkpoint_manager, load_checkpoint, save_checkpoint
    from xtrax.engine.engine import Engine
    from xtrax.training.grad import accumulate_grads
    from xtrax.training.state import init_state
    from xtrax.training.step import create_train_step
    from xtrax.training.trainer import Trainer


    def loss(model, batch, key):
        return jnp.mean((batch["x"] @ model["w"] + model["b"] - batch["y"]) ** 2)


    class Data:
        def train_iter(self):
            return iter([_batch(i) for i in range(3)])

        def eval_iter(self):
            return iter([_batch(9)])


    def _batch(i):
        return {"x": jnp.ones((4, 3)) * (i + 1), "y": jnp.ones((4, 2))}


    model = {"w": jnp.ones((3, 2)) * 0.1, "b": jnp.zeros(2), "tag": "dict"}
    opt = optax.sgd(0.01)
    trainer = Trainer(loss_fn=loss, optimizer=opt, takes_key=True)
    state = init_state(model, opt, 0)
    for i in range(3):
        state, metrics = trainer.step(state, _batch(i))
    assert int(state.step) == 3 and state.model["tag"] == "dict"

    stacked = {"x": jnp.ones((2, 4, 3)), "y": jnp.ones((2, 4, 2))}
    grads, mean_loss = accumulate_grads(
        lambda p, b: loss(p, b, None), {"w": model["w"], "b": model["b"]}, stacked
    )
    assert grads["w"].shape == (3, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        safe = create_train_step(loss, opt, safety=True, takes_key=True)
    arrays_only = {"w": model["w"], "b": model["b"]}
    safe_state, _ = safe.step(init_state(arrays_only, opt, 0), _batch(0))
    assert int(safe_state.step) == 1

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["XTRAX_LEDGER_ROOT"] = str(pathlib.Path(tmp) / "ledger")
        engine = Engine(trainer=trainer, callbacks=())
        ckpt = pathlib.Path(tmp) / "ckpt"
        start = init_state(model, opt, 0)
        final = engine.fit_sync(start, Data(), num_epochs=1, checkpoint_dir=ckpt)
        assert int(final.step) == 3
        evaluated = asyncio.run(engine.eval(final, Data()))
        assert "loss" in evaluated
        restored = engine.restore(ckpt, init_state(model, opt, 1))
        assert int(restored.step) == 3

    assert "equinox" not in sys.modules


try:
    exercise()
    exercise_error = None
except Exception as exc:
    exercise_error = f"{type(exc).__name__}: {exc}"

report = {"touched": sorted(touched), "n_modules": len(names), "exercise_error": exercise_error}
print(json.dumps(report))
"""


def _run_gate(inject=None):
    code = _GATE.replace("INJECT_TARGET", repr(inject))
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _unexpected(report):
    return sorted(set(report["touched"]) - _PENDING)


def test_xtrax_and_the_training_path_never_import_equinox():
    report = _run_gate()
    assert report["n_modules"] > 150  # the walk really covered the package
    assert _unexpected(report) == []
    assert report["exercise_error"] is None


def test_pending_list_is_not_stale():
    report = _run_gate()
    assert _PENDING - set(report["touched"]) == set(), "delete the migrated entries from _PENDING"


def test_gate_control_an_injected_import_equinox_goes_red():
    # prove red: one real training module gains `import equinox`; the same gate must flag it
    leaked = _run_gate(inject="xtrax.training.trainer")
    unexpected = _unexpected(leaked)
    assert "xtrax.training.trainer" in unexpected
    # and everything that imports the trainer is flagged with it (engine, step, cli verbs...)
    assert "xtrax.engine.engine" in unexpected and "xtrax.training.step" in unexpected
    # the dict-model train, which imports the trainer, cannot run either
    assert "equinox import blocked" in leaked["exercise_error"]


def test_gate_control_the_blocker_itself_raises():
    code = _GATE.split("sys.meta_path.insert(0, Block())")[0] + (
        "sys.meta_path.insert(0, Block())\nimport equinox\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "equinox import blocked" in result.stderr


# --- AST sweep over the modules this step migrated -----------------------------------------

_MIGRATED = [
    "training/__init__.py",
    "training/grad.py",
    "training/loss.py",
    "training/state.py",
    "training/step.py",
    "training/trainer.py",
    "training/types.py",
    "engine/engine.py",
    "safety/manager.py",
    "distributed/sharding.py",
    "telemetry/ir.py",
    "devtools/gates/_performance_probes.py",
    "devtools/gates/_trace_probe.py",
]


def _equinox_references(source):
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        elif isinstance(node, ast.Name) and node.id == "eqx":
            names.append("eqx")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "equinox":
                names.append("equinox")
    return [n for n in names if n.split(".")[0] in {"equinox", "eqx"}]


@pytest.mark.parametrize("rel", _MIGRATED)
def test_migrated_module_has_no_equinox_import_or_eqx_reference(rel):
    assert _equinox_references((_SRC / rel).read_text()) == []


@pytest.mark.parametrize(
    "bad",
    [
        "import equinox",
        "import equinox as eqx",
        "from equinox import nn",
        "import jax\n\ndef f():\n    import equinox as eqx\n    return eqx.is_array",
        "x = eqx.filter_jit",
    ],
)
def test_ast_sweep_control_flags_an_equinox_reference(bad):
    assert _equinox_references(bad) != []
