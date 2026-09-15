"""xtrax.export.rings: toolchain-gated real tests, plus fake-injected logic tests.

Every AC referenced in a test name/comment is quoted from
``.praxia/docs/specs/260911_export-divergence-mapping.md`` section 10.

Two kinds of test live here, per the module's own coverage split:

- **Fake-injected** tests never touch IREE. Most of ``rings.py``'s own logic
  (name recovery, input-class generators, ring orchestration, gating) is pure
  JAX or pure Python, and is monkeypatched at the ``compile_for_target`` /
  ``run_native_vmfb`` boundary -- the only place IREE actually enters. These
  run in every environment, including ``dev``+``io`` with no ``export`` extra.
- **Toolchain-gated** tests (``pytest.importorskip("iree.compiler")`` /
  ``("iree.runtime")``, matching ``test_compile_native_wasm32.py`` /
  ``test_parity_multi_size.py``) exercise the real compiled path. They skip
  cleanly wherever IREE is absent.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export import divergence as d
from xtrax.export import rings
from xtrax.export.composer import build_traceable_callable

# ---------------------------------------------------------------------------
# T8 -- section 6.2 input-class generators (AC-17)
# ---------------------------------------------------------------------------


class TestBucketLadder:
    def test_is_the_documented_tuple(self):
        assert rings.BUCKET_LADDER == (64, 128, 256, 512, 1024, 1536, 2048)


class TestInputClassGenerators:
    def test_nominal_labels_the_callers_own_input(self):
        """`nominal` is a thin, labelled pass-through -- xtrax cannot manufacture
        a downstream model's own reference input.
        """
        ai = (jax.ShapeDtypeStruct((3,), jnp.float32),)
        ci = (jnp.zeros(3),)
        result = rings.nominal(ai, ci)
        assert result.label == "nominal"
        assert result.in_contract is True
        assert result.abstract_inputs == ai
        assert result.concrete_inputs == ci

    def test_symmetric_geometry_is_in_contract_and_all_ones_mask(self):
        """AC-17: symmetric_geometry is the primary, in-contract class (M6)."""
        result = rings.symmetric_geometry(64)
        assert result.label == "symmetric_geometry"
        assert result.in_contract is True
        coords, mask = result.concrete_inputs
        assert coords.shape == (64, 3)
        assert bool(jnp.all(mask))

    def test_symmetric_geometry_produces_exact_geometric_ties(self):
        """An ideal helix: pairs at equal |i-j| separation are exactly tied."""
        result = rings.symmetric_geometry(64)
        coords, _mask = result.concrete_inputs
        d0 = float(jnp.linalg.norm(coords[0] - coords[2]))
        d1 = float(jnp.linalg.norm(coords[1] - coords[3]))
        assert d0 == pytest.approx(d1, abs=1e-4)

    def test_symmetric_geometry_rejects_a_non_bucket_length(self):
        """Section 6.2: length is not a free axis."""
        with pytest.raises(ValueError, match="bucket ladder"):
            rings.symmetric_geometry(100)

    def test_magnitude_extremes_is_in_contract_and_finite(self):
        result = rings.magnitude_extremes(64)
        assert result.label == "magnitude_extremes"
        assert result.in_contract is True
        coords, mask = result.concrete_inputs
        assert bool(jnp.all(mask))
        assert np.isfinite(np.asarray(coords)).all()

    def test_magnitude_extremes_spans_near_overflow_and_near_denormal(self):
        result = rings.magnitude_extremes(64)
        coords, _mask = result.concrete_inputs
        magnitudes = np.abs(np.asarray(coords))
        finfo = np.finfo(np.float32)
        assert magnitudes.max() > finfo.max * 0.05
        assert magnitudes[magnitudes > 0].min() < finfo.tiny * 1e5

    def test_sub_k_neighbours_is_labelled_out_of_contract(self):
        """AC-17: sub_k_neighbours must never carry a verdict alone."""
        result = rings.sub_k_neighbours(64, k_neighbors=48)
        assert result.label == "sub_k_neighbours"
        assert result.in_contract is False

    def test_sub_k_neighbours_mask_is_actually_below_the_clamp(self):
        result = rings.sub_k_neighbours(64, k_neighbors=48)
        _coords, mask = result.concrete_inputs
        assert int(jnp.sum(mask)) < 48

    def test_sub_k_neighbours_rejects_a_non_bucket_length(self):
        with pytest.raises(ValueError, match="bucket ladder"):
            rings.sub_k_neighbours(100, k_neighbors=48)


# ---------------------------------------------------------------------------
# T7 -- probe-name recovery from out_tree (AC-2)
# ---------------------------------------------------------------------------


class TestRecoverProbeNames:
    def test_dict_sorted_order_differs_from_declaration_order(self):
        """AC-2: a dict whose sorted key order differs from declaration order."""

        def f(x):
            return {"zeta": x + 1, "alpha": x * 2, "mid": x * 3}

        exported = jax.export.export(jax.jit(f))(jax.ShapeDtypeStruct((3,), jnp.float32))
        # The runtime returns flat values in out_tree (sorted-key) order:
        # alpha, mid, zeta -- NOT declaration order (zeta, alpha, mid).
        flat = [jnp.full((3,), 9.0), jnp.full((3,), 1.0), jnp.full((3,), 5.0)]
        recovered = rings.recover_probe_names(exported.out_tree, flat)
        assert set(recovered) == {"alpha", "mid", "zeta"}
        assert np.allclose(recovered["alpha"], 9.0)
        assert np.allclose(recovered["mid"], 1.0)
        assert np.allclose(recovered["zeta"], 5.0)

    def test_namedtuple_field_order_differs_from_sorted_order(self):
        """AC-2: a NamedTuple whose field order differs from sorted order."""

        class Out(NamedTuple):
            zeta: jax.Array
            alpha: jax.Array
            mid: jax.Array

        def f(x):
            return Out(zeta=x + 1, alpha=x * 2, mid=x * 3)

        exported = jax.export.export(jax.jit(f))(jax.ShapeDtypeStruct((3,), jnp.float32))
        # The runtime returns flat values in out_tree (field) order:
        # zeta, alpha, mid -- NOT sorted order (alpha, mid, zeta).
        flat = [jnp.full((3,), 9.0), jnp.full((3,), 1.0), jnp.full((3,), 5.0)]
        recovered = rings.recover_probe_names(exported.out_tree, flat)
        assert set(recovered) == {"zeta", "alpha", "mid"}
        assert np.allclose(recovered["zeta"], 9.0)
        assert np.allclose(recovered["alpha"], 1.0)
        assert np.allclose(recovered["mid"], 5.0)

    def test_bare_array_output_recovers_the_empty_bare_name(self):
        def f(x):
            return x + 1

        exported = jax.export.export(jax.jit(f))(jax.ShapeDtypeStruct((3,), jnp.float32))
        recovered = rings.recover_probe_names(exported.out_tree, [jnp.full((3,), 7.0)])
        assert set(recovered) == {""}
        assert np.allclose(recovered[""], 7.0)


class TestProbeResultPaths:
    """Coordinator-confirmed conventions: dict -> "result['name']", NamedTuple -> "result.name"."""

    def test_dict_form(self):
        def f(x):
            return {"rbf": x + 1, "final": x * 2}

        exported = jax.export.export(jax.jit(f))(jax.ShapeDtypeStruct((3,), jnp.float32))
        paths = rings._probe_result_paths(exported.out_tree)
        assert paths == {"final": "result['final']", "rbf": "result['rbf']"}

    def test_namedtuple_form(self):
        class Out(NamedTuple):
            rbf: jax.Array
            final: jax.Array

        def f(x):
            return Out(rbf=x + 1, final=x * 2)

        exported = jax.export.export(jax.jit(f))(jax.ShapeDtypeStruct((3,), jnp.float32))
        paths = rings._probe_result_paths(exported.out_tree)
        assert paths == {"rbf": "result.rbf", "final": "result.final"}

    def test_bare_array_form(self):
        def f(x):
            return x + 1

        exported = jax.export.export(jax.jit(f))(jax.ShapeDtypeStruct((3,), jnp.float32))
        paths = rings._probe_result_paths(exported.out_tree)
        assert paths == {"": "result"}


# ---------------------------------------------------------------------------
# AC-11 -- R0/R1/R2 runners accept no probe argument (API-surface claim)
# ---------------------------------------------------------------------------


class TestAC11NoProbeArgumentOnR0R1R2:
    @pytest.mark.parametrize(
        "fn",
        [rings.r0_replay_gate, rings.r1_target_isa, rings.r2a_fusion, rings.r2b_lowering],
    )
    def test_signature_has_no_probe_shaped_parameter(self, fn):
        params = list(inspect.signature(fn).parameters)
        assert not any("probe" in name for name in params), (
            f"{fn.__name__} has a probe-shaped parameter: {params}"
        )

    def test_r3_is_the_one_exception(self):
        params = list(inspect.signature(rings.r3_probe).parameters)
        assert "probe_deps" in params


# ---------------------------------------------------------------------------
# Shared fake fixtures (fake-injected, no toolchain)
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_plan():
    """A single-axis Vmap plan over 4 elements of shape (2,)."""
    from xtrax.tiling.plan import AxisDecision, AxisSpec
    from xtrax.tiling.strategy import Vmap

    class _Plan:
        def __init__(self, decisions):
            self.decisions = decisions

    return _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name="batch", cardinality=4, default_batch_size=0),
                batch_size=0,
                reasoning="rings test fixture",
                strategy=Vmap(),
            )
        ]
    )


@pytest.fixture
def toy_xs():
    return jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2) / 8.0


@pytest.fixture
def toy_abstract_inputs(toy_xs):
    return [jax.ShapeDtypeStruct(toy_xs.shape, toy_xs.dtype)]


def _bare_fn(x):
    """A single-array-output per-element function; exercises the "" bare name path."""
    return x + 1.0


def _named_fn(x):
    """A two-name dict-output per-element function: `probe1` (no preds) -> `final`."""
    return {"probe1": x * 2.0, "final": x + 1.0}


_NAMED_PROBE_DEPS = {"probe1": (), "final": ("probe1",)}


def _chain_fn(x):
    """Mirrors spec SS6.1's own example shape: a SOURCE probe
    (`neighbor_indices`) that appears ONLY as somebody else's predecessor --
    never as its own `probe_deps` key.
    """
    return {"neighbor_indices": x, "rbf": x * 2.0, "final": x + 1.0}


# Same shape as spec SS6.1's `{"rbf": ("neighbor_indices",), ...}`:
# "neighbor_indices" has no entry of its own in this mapping.
_CHAIN_PROBE_DEPS = {"rbf": ("neighbor_indices",), "final": ("rbf",)}


# ---------------------------------------------------------------------------
# T4 -- R0 gate (AC-16)
# ---------------------------------------------------------------------------


class TestR0ReplayGate:
    def test_deterministic_replays_pass(self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs):
        fake_path = Path("/fake/r0.vmfb")
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=fake_path),
        )
        fixed = jnp.ones((4, 2), dtype=jnp.float32)
        monkeypatch.setattr(rings, "run_native_vmfb", lambda path, *a, function="main": fixed)

        result = rings.r0_replay_gate(_bare_fn, toy_plan, toy_abstract_inputs, (toy_xs,), replays=3)
        assert result.ring == "R0"
        assert result.passed is True
        assert result.probes == ()
        assert result.input_class == "nominal"
        assert result.in_contract is True

    def test_nondeterministic_replay_is_a_hard_gate(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """AC-16: a deliberately nondeterministic fake artifact prevents any ring from running."""
        fake_path = Path("/fake/r0.vmfb")
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=fake_path),
        )
        sequence = iter(
            [
                jnp.ones((4, 2), dtype=jnp.float32),
                jnp.ones((4, 2), dtype=jnp.float32),
                jnp.zeros((4, 2), dtype=jnp.float32),  # replay 2 disagrees
            ]
        )
        monkeypatch.setattr(
            rings, "run_native_vmfb", lambda path, *a, function="main": next(sequence)
        )

        result = rings.r0_replay_gate(_bare_fn, toy_plan, toy_abstract_inputs, (toy_xs,), replays=3)
        assert result.ring == "R0"
        assert result.passed is False
        assert any("replay 2" in n for n in result.notes)


# ---------------------------------------------------------------------------
# T4 -- R1 (AC-16)
# ---------------------------------------------------------------------------


class TestR1TargetIsa:
    def test_refuses_to_interpret_when_cpu_features_are_identical(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """AC-16: R1 refuses to interpret its legs when cpu_features readback matches."""
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(
                path=Path(f"/fake/{target.name}.vmfb")
            ),
        )
        monkeypatch.setattr(rings, "_read_cpu_features", lambda path: "same-features")

        result = rings.r1_target_isa(_bare_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        assert result.ring == "R1"
        assert result.passed is False
        assert result.probes == ()
        assert any("refuses to interpret" in n for n in result.notes)

    def test_interprets_legs_when_cpu_features_differ(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(
                path=Path(f"/fake/{target.name}.vmfb")
            ),
        )

        def fake_features(path: Path) -> str:
            return {"native.vmfb": "AAA", "native-portable.vmfb": "BBB"}[path.name]

        monkeypatch.setattr(rings, "_read_cpu_features", fake_features)

        def fake_run(path: Path, *args, function="main"):
            return jnp.ones((4, 2), dtype=jnp.float32)

        monkeypatch.setattr(rings, "run_native_vmfb", fake_run)

        result = rings.r1_target_isa(_bare_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        assert result.ring == "R1"
        assert result.passed is True
        assert result.probes == ()
        assert any("AAA" in n for n in result.notes)
        assert any("BBB" in n for n in result.notes)


# ---------------------------------------------------------------------------
# T5 -- R2a fusion + section 3.2 budget derivation
# ---------------------------------------------------------------------------


class TestR2aFusion:
    def test_bare_output_produces_a_budget_at_the_floor(
        self, toy_plan, toy_xs, toy_abstract_inputs
    ):
        def eager_fn(inputs):
            (arr,) = inputs
            return jnp.stack([_bare_fn(arr[i]) for i in range(arr.shape[0])])

        result, budgets = rings.r2a_fusion(
            _bare_fn, toy_plan, toy_abstract_inputs, (toy_xs,), eager_fn=eager_fn
        )
        assert result.ring == "R2a"
        assert result.passed is True
        assert result.probes == ()
        # bare-array output -> top-level name "" -> budget_key("", "") == ""
        assert "" in budgets
        assert budgets[""] == pytest.approx(d.ULP_FLOOR)

    def test_named_output_keys_budgets_per_name(self, toy_plan, toy_xs, toy_abstract_inputs):
        def eager_fn(inputs):
            (arr,) = inputs
            return {
                "probe1": jnp.stack([arr[i] * 2.0 for i in range(arr.shape[0])]),
                "final": jnp.stack([arr[i] + 1.0 for i in range(arr.shape[0])]),
            }

        result, budgets = rings.r2a_fusion(
            _named_fn, toy_plan, toy_abstract_inputs, (toy_xs,), eager_fn=eager_fn
        )
        assert result.passed is True
        assert set(budgets) == {"probe1", "final"}
        assert budgets["probe1"] == pytest.approx(d.ULP_FLOOR)
        assert budgets["final"] == pytest.approx(d.ULP_FLOOR)
        assert d.budget_key("final", "") == "final"

    def test_a_probe_eager_fn_omits_still_gets_a_budget(
        self, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """FINDING 5: `eager_fn` is documented as "the model applied directly"
        (section 5.1) -- a real caller's oracle normally covers only the true
        model output(s), not pipeline-internal probes. A probe name absent
        from it must not be silently skipped (leaving it with no budget, so a
        later non-identical R3 leaf raises `MissingBudgetError`/`KeyError`
        only after every compile has already run) -- R2a must measure it too,
        by running the SAME callable eagerly (un-jitted) itself, since no
        separate "independent oracle" is possible for a pipeline
        intermediate.
        """

        def eager_fn(inputs):
            (arr,) = inputs
            # Only "final" -- omits "probe1" entirely, exactly like a real
            # caller's model-level reference implementation would.
            return {"final": jnp.stack([arr[i] + 1.0 for i in range(arr.shape[0])])}

        result, budgets = rings.r2a_fusion(
            _named_fn, toy_plan, toy_abstract_inputs, (toy_xs,), eager_fn=eager_fn
        )
        assert result.passed is True
        assert "probe1" in budgets, (
            "probe1 was absent from eager_fn's output but must still receive "
            "a calibrated budget -- R2a must not silently skip it"
        )
        assert budgets["probe1"] == pytest.approx(d.ULP_FLOOR)
        assert not any("skipped" in n for n in result.notes)


# ---------------------------------------------------------------------------
# T5 -- R2b lowering
# ---------------------------------------------------------------------------


class TestR2bLowering:
    def test_reports_jit_vs_iree_leaves(self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs):
        fake_path = Path("/fake/r2b.vmfb")
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=fake_path),
        )
        monkeypatch.setattr(
            rings,
            "run_native_vmfb",
            lambda path, *a, function="main": jnp.zeros((4, 2), dtype=jnp.float32),
        )

        result = rings.r2b_lowering(_bare_fn, toy_plan, toy_abstract_inputs, (toy_xs,), budgets={})
        assert result.ring == "R2b"
        assert result.passed is True
        assert result.probes == ()
        assert len(result.notes) == 1


# ---------------------------------------------------------------------------
# T6 -- R3 probe rung, with the section 5.4 fidelity precondition (AC-6)
# ---------------------------------------------------------------------------


def _flatten_for(fn, plan, abstract_inputs, concrete_inputs):
    callable_ = build_traceable_callable(fn, plan, None)
    return list(jax.tree_util.tree_leaves(jax.jit(callable_)(*concrete_inputs)))


class TestR3Probe:
    def test_refuses_when_r0_failed(self):
        """section 5.4: if R0 fails, exact equality is unachievable; R3 must not run."""
        r0_result = d.RingResult(
            ring="R0", passed=False, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        result = rings.r3_probe(
            _bare_fn,
            plan=None,
            abstract_inputs=(),
            concrete_inputs=(),
            probe_deps={},
            budgets={},
            r0_result=r0_result,
        )
        assert result.ring == "R3"
        assert result.passed is False
        assert result.probes == ()
        assert any("R0" in n for n in result.notes)

    def test_happy_path_classifies_every_declared_probe(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        primary_fn = rings._primary_only(_named_fn, frozenset({"probe1"}))
        primary_flat = _flatten_for(primary_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        full_flat = _flatten_for(_named_fn, toy_plan, toy_abstract_inputs, (toy_xs,))

        call_paths = [Path("/fake/primary.vmfb"), Path("/fake/full.vmfb")]
        compile_calls = iter(call_paths)
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=next(compile_calls)),
        )
        outputs = {call_paths[0]: tuple(primary_flat), call_paths[1]: tuple(full_flat)}
        monkeypatch.setattr(
            rings, "run_native_vmfb", lambda path, *a, function="main": outputs[path]
        )

        r0_result = d.RingResult(
            ring="R0", passed=True, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        result = rings.r3_probe(
            _named_fn,
            toy_plan,
            toy_abstract_inputs,
            (toy_xs,),
            _NAMED_PROBE_DEPS,
            budgets={},
            r0_result=r0_result,
        )
        assert result.ring == "R3"
        assert result.passed is True
        assert {p.name for p in result.probes} == {"probe1", "final"}
        # Every leaf is bit-identical (fake IREE == real jit): both probes CLEAN.
        assert all(p.divergence_class == d.DivergenceClass.CLEAN for p in result.probes)

    def _corrupted_fidelity_result(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs, corrupt
    ):
        primary_fn = rings._primary_only(_named_fn, frozenset({"probe1"}))
        primary_flat = _flatten_for(primary_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        full_flat = corrupt(_flatten_for(_named_fn, toy_plan, toy_abstract_inputs, (toy_xs,)))

        call_paths = [Path("/fake/primary.vmfb"), Path("/fake/full.vmfb")]
        compile_calls = iter(call_paths)
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=next(compile_calls)),
        )
        outputs = {call_paths[0]: tuple(primary_flat), call_paths[1]: tuple(full_flat)}
        monkeypatch.setattr(
            rings, "run_native_vmfb", lambda path, *a, function="main": outputs[path]
        )

        r0_result = d.RingResult(
            ring="R0", passed=True, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        # A real (small) budget for "final", as R2a would have produced --
        # the corruption below is orders of magnitude past it either way.
        return rings.r3_probe(
            _named_fn,
            toy_plan,
            toy_abstract_inputs,
            (toy_xs,),
            _NAMED_PROBE_DEPS,
            budgets={"final": d.ULP_FLOOR},
            r0_result=r0_result,
        )

    def test_ac6_instrumentation_changed_result_suppression(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """AC-6, fixture 1: instrumentation appears to *suppress* the primary output."""

        def corrupt(flat):
            # dict flattens sorted-key: "final" < "probe1", so index 0 is "final".
            flat[0] = flat[0] + 100.0
            return flat

        result = self._corrupted_fidelity_result(
            monkeypatch, toy_plan, toy_xs, toy_abstract_inputs, corrupt
        )
        assert result.ring == "R3"
        assert result.passed is False
        assert result.probes == ()
        assert any("INSTRUMENTATION_CHANGED_RESULT" in n for n in result.notes)

    def test_ac6_instrumentation_changed_result_creation(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """AC-6, fixture 2: same branch, different fixture (revision-5 note on "two-sided")."""

        def corrupt(flat):
            flat[0] = flat[0] * 0.0 - 999.0
            return flat

        result = self._corrupted_fidelity_result(
            monkeypatch, toy_plan, toy_xs, toy_abstract_inputs, corrupt
        )
        assert result.ring == "R3"
        assert result.passed is False
        assert result.probes == ()
        assert any("INSTRUMENTATION_CHANGED_RESULT" in n for n in result.notes)

    def test_no_sink_raises(self):
        """A cyclic-looking declaration with no sink cannot yield a primary output."""
        r0_result = d.RingResult(
            ring="R0", passed=True, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        with pytest.raises(ValueError, match="sink"):
            rings.r3_probe(
                _named_fn,
                plan=None,
                abstract_inputs=(),
                concrete_inputs=(),
                probe_deps={"a": ("b",), "b": ("a",)},
                budgets={},
                r0_result=r0_result,
            )

    def test_source_probe_declared_only_as_a_predecessor_is_not_dropped(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """FINDING 2: spec SS6.1's own `probe_deps` shape must not crash R3.

        `neighbor_indices` is a source probe with no predecessors of its own
        -- it appears ONLY as `rbf`'s predecessor, so it is never a
        `probe_deps` key. The probe set R3 strips/classifies over must be the
        union of the mapping's keys and every predecessor name, not just the
        keys -- otherwise `neighbor_indices` is never stripped from the
        "uninstrumented" (primary-only) view, `_compare_by_name` raises a
        top-level name mismatch, and (if that were bypassed) `classify_probes`
        would separately raise on an "undeclared predecessor".
        """
        primary_fn = rings._primary_only(_chain_fn, frozenset({"neighbor_indices", "rbf"}))
        primary_flat = _flatten_for(primary_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        full_flat = _flatten_for(_chain_fn, toy_plan, toy_abstract_inputs, (toy_xs,))

        call_paths = [Path("/fake/primary.vmfb"), Path("/fake/full.vmfb")]
        compile_calls = iter(call_paths)
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=next(compile_calls)),
        )
        outputs = {call_paths[0]: tuple(primary_flat), call_paths[1]: tuple(full_flat)}
        monkeypatch.setattr(
            rings, "run_native_vmfb", lambda path, *a, function="main": outputs[path]
        )

        r0_result = d.RingResult(
            ring="R0", passed=True, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        result = rings.r3_probe(
            _chain_fn,
            toy_plan,
            toy_abstract_inputs,
            (toy_xs,),
            _CHAIN_PROBE_DEPS,
            budgets={},
            r0_result=r0_result,
        )
        assert result.ring == "R3"
        assert result.passed is True
        # The source probe must appear in the report, not be silently dropped.
        assert {p.name for p in result.probes} == {"neighbor_indices", "rbf", "final"}
        assert all(p.divergence_class == d.DivergenceClass.CLEAN for p in result.probes)

    def test_report_order_does_not_depend_on_all_probe_names_iteration_order(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """FINDING 5 (round 3): R3's probe report order must not depend on
        `PYTHONHASHSEED`.

        `probe_leaves` used to be filled by iterating `_all_probe_names`'s
        return value directly -- a `frozenset[str]`, whose iteration order
        depends on Python's per-process string hash randomization -- and
        `classify_probes` preserves whatever order it was built in (its own
        docstring: "in probes' iteration order"). Two different (but
        set-equal) orderings of the same probe names, forced here by
        monkeypatching `_all_probe_names`, must not change the resulting
        report order.
        """

        class _ReversedIterationOrder(frozenset):
            def __iter__(self):
                return reversed(list(super().__iter__()))

        names = {"neighbor_indices", "rbf", "final"}
        forward = frozenset(names)
        backward = _ReversedIterationOrder(names)
        assert tuple(forward) != tuple(backward), (
            "test fixture bug: the two orderings must actually differ"
        )

        primary_fn = rings._primary_only(_chain_fn, frozenset({"neighbor_indices", "rbf"}))
        primary_flat = _flatten_for(primary_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        full_flat = _flatten_for(_chain_fn, toy_plan, toy_abstract_inputs, (toy_xs,))

        def _run(order_value):
            call_paths = [Path("/fake/primary.vmfb"), Path("/fake/full.vmfb")]
            compile_calls = iter(call_paths)
            monkeypatch.setattr(
                rings,
                "compile_for_target",
                lambda mlir, target, out_path=None: SimpleNamespace(path=next(compile_calls)),
            )
            outputs = {call_paths[0]: tuple(primary_flat), call_paths[1]: tuple(full_flat)}
            monkeypatch.setattr(
                rings, "run_native_vmfb", lambda path, *a, function="main": outputs[path]
            )
            monkeypatch.setattr(rings, "_all_probe_names", lambda probe_deps: order_value)

            r0_result = d.RingResult(
                ring="R0",
                passed=True,
                input_class="nominal",
                in_contract=True,
                probes=(),
                notes=(),
            )
            result = rings.r3_probe(
                _chain_fn,
                toy_plan,
                toy_abstract_inputs,
                (toy_xs,),
                _CHAIN_PROBE_DEPS,
                budgets={},
                r0_result=r0_result,
            )
            return tuple(p.name for p in result.probes)

        order_a = _run(forward)
        order_b = _run(backward)
        assert order_a == order_b, (
            f"probe report order depends on _all_probe_names' own iteration "
            f"order: {order_a} != {order_b}"
        )


def _aux_fn(x):
    """Round-3 FINDING 2 reproducer: an undeclared top-level output (`aux`)
    alongside a two-probe chain.
    """
    return {
        "final": x + 1.0,
        "aux": x * 3.0,
        "rbf": x * 2.0,
        "neighbor_indices": x,
    }


_AUX_PROBE_DEPS = {"rbf": ("neighbor_indices",), "final": ("rbf",)}


class TestR3ProbeUndeclaredOutputFidelityFilter:
    def test_undeclared_output_does_not_crash_the_fidelity_check(
        self, monkeypatch, toy_plan, toy_xs, toy_abstract_inputs
    ):
        """FINDING 2 (round 3): `primary_actual` and `instrumented_primary` must
        use the SAME name filter.

        `primary_actual` keeps every top-level name NOT in `strip_names` (so
        an undeclared output like `aux` -- never mentioned in `probe_deps` at
        all -- stays); `instrumented_primary` used to keep only names IN
        `sinks`, dropping `aux`. `_compare_by_name` then raised a top-level
        name mismatch (`only in expected=['aux']`) after both artifacts had
        already compiled and run.
        """
        primary_fn = rings._primary_only(_aux_fn, frozenset({"neighbor_indices", "rbf"}))
        primary_flat = _flatten_for(primary_fn, toy_plan, toy_abstract_inputs, (toy_xs,))
        full_flat = _flatten_for(_aux_fn, toy_plan, toy_abstract_inputs, (toy_xs,))

        call_paths = [Path("/fake/primary.vmfb"), Path("/fake/full.vmfb")]
        compile_calls = iter(call_paths)
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=next(compile_calls)),
        )
        outputs = {call_paths[0]: tuple(primary_flat), call_paths[1]: tuple(full_flat)}
        monkeypatch.setattr(
            rings, "run_native_vmfb", lambda path, *a, function="main": outputs[path]
        )

        r0_result = d.RingResult(
            ring="R0", passed=True, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        result = rings.r3_probe(
            _aux_fn,
            toy_plan,
            toy_abstract_inputs,
            (toy_xs,),
            _AUX_PROBE_DEPS,
            budgets={},
            r0_result=r0_result,
        )
        assert result.ring == "R3"
        assert result.passed is True
        # `aux` is undeclared -- neither a probe nor a sink -- and correctly
        # absent from the classified probe report; only the declared names
        # are classified.
        assert {p.name for p in result.probes} == {"neighbor_indices", "rbf", "final"}
        assert all(p.divergence_class == d.DivergenceClass.CLEAN for p in result.probes)


def _scan_plan(lane_n: int, step_n: int):
    """A real Vmap-over-Scan plan (the certified two-axis shape)."""
    from xtrax.tiling.plan import AxisDecision, AxisSpec
    from xtrax.tiling.strategy import Scan, Vmap

    class _Plan:
        def __init__(self, decisions):
            self.decisions = decisions

    return _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name="lane", cardinality=lane_n, default_batch_size=0),
                batch_size=0,
                reasoning="rings test fixture",
                strategy=Vmap(),
            ),
            AxisDecision(
                spec=AxisSpec(name="step", cardinality=step_n, default_batch_size=0),
                batch_size=0,
                reasoning="rings test fixture",
                strategy=Scan(init=None),
            ),
        ]
    )


def _scan_transition(carry, x):
    """A Scan transition returning `(carry, y)`, with named probes inside `y`."""
    carry = carry + x
    y = {"probe1": carry * 2.0, "final": carry + 1.0}
    return carry, y


_SCAN_PROBE_DEPS = {"probe1": (), "final": ("probe1",)}


class TestR3ProbeOverAScanPlan:
    def test_runs_over_a_real_vmap_of_scan_plan(self, monkeypatch):
        """FINDING 1 (round 3): R3 must not crash on a Scan-based plan.

        `_primary_only` used to wrap the per-element `fn` BEFORE it went into
        `build_traceable_callable`. For a Scan axis -- including this
        certified Vmap-over-Scan shape -- `fn` is a TRANSITION returning
        `(carry, y)`, not a plain per-element function; stripping its return
        value turns that 2-tuple into a one-key dict `{"": (carry, y)}`,
        which the composer's own `_batched_transition` then fails to unpack
        as `carry, y = fn(carry, x)` (composer.py:276) -- at TRACE time,
        before IREE ever runs. The strip must happen on the COMPOSED
        callable's own output instead, after `build_traceable_callable` has
        already folded the scan.
        """
        plan = _scan_plan(lane_n=2, step_n=3)
        init = jnp.zeros((2,), dtype=jnp.float32)
        xs = jnp.arange(3, dtype=jnp.float32)
        abstract_inputs = [jax.ShapeDtypeStruct(xs.shape, xs.dtype)]

        # Build the expected fixture values by composing FIRST, then
        # stripping the composed callable's own output -- the fix under
        # test, used correctly here to produce the ground truth this
        # reproducer checks r3_probe against.
        full_callable = build_traceable_callable(_scan_transition, plan, None, scan_init=init)
        primary_callable = rings._primary_only(full_callable, frozenset({"probe1"}))
        primary_flat = list(jax.tree_util.tree_leaves(jax.jit(primary_callable)(xs)))
        full_flat = list(jax.tree_util.tree_leaves(jax.jit(full_callable)(xs)))

        call_paths = [Path("/fake/primary.vmfb"), Path("/fake/full.vmfb")]
        compile_calls = iter(call_paths)
        monkeypatch.setattr(
            rings,
            "compile_for_target",
            lambda mlir, target, out_path=None: SimpleNamespace(path=next(compile_calls)),
        )
        outputs = {call_paths[0]: tuple(primary_flat), call_paths[1]: tuple(full_flat)}
        monkeypatch.setattr(
            rings, "run_native_vmfb", lambda path, *a, function="main": outputs[path]
        )

        r0_result = d.RingResult(
            ring="R0", passed=True, input_class="nominal", in_contract=True, probes=(), notes=()
        )
        result = rings.r3_probe(
            _scan_transition,
            plan,
            abstract_inputs,
            (xs,),
            _SCAN_PROBE_DEPS,
            budgets={},
            r0_result=r0_result,
            scan_init=init,
        )
        assert result.ring == "R3"
        assert result.passed is True
        assert {p.name for p in result.probes} == {"probe1", "final"}
        assert all(p.divergence_class == d.DivergenceClass.CLEAN for p in result.probes)


class TestR2aFusionScanEagerFallback:
    """FINDING 3 (round 4): R2a's "eager" fallback for a probe `eager_fn`
    omits is not actually eager for a Scan-based composition.

    `lax.scan` always compiles its whole body into a single XLA computation
    -- even outside `jax.jit` -- so a naive un-jitted call to the composed
    callable traces the transition's Python body exactly ONCE (at trace
    time), not once per step. Proven with a Python-level side effect: under
    genuinely eager (per-step) execution an N-step scan runs the transition
    body N times in the Python interpreter; under compiled scan (jit OR a
    naive un-jitted call) it runs once.
    """

    def test_naive_unjitted_call_traces_the_scan_body_once_not_once_per_step(self):
        """Establishes the bug is real, independent of r2a_fusion, before
        trusting any assertion about r2a_fusion's own behavior.
        """
        step_n = 4
        calls: list[int] = []

        def transition(carry, x):
            calls.append(1)
            carry = carry + x
            return carry, carry

        plan = _scan_plan(lane_n=1, step_n=step_n)
        init = jnp.zeros((1,), dtype=jnp.float32)
        xs = jnp.arange(step_n, dtype=jnp.float32)
        callable_ = build_traceable_callable(transition, plan, None, scan_init=init)

        calls.clear()
        callable_(xs)
        assert len(calls) == 1, (
            "a naive un-jitted call to a Scan-composed callable is expected to "
            "trace the transition body once (lax.scan compiles regardless of "
            "jax.jit) -- if this fails, lax.scan's behavior in this jax "
            "version has changed and FINDING 3 needs re-evaluating"
        )

        calls.clear()
        with jax.disable_jit():
            callable_(xs)
        assert len(calls) == step_n, (
            "jax.disable_jit() is expected to force lax.scan to execute its "
            "body once per step in the Python interpreter in this jax "
            "version -- if this fails, disable_jit does not achieve true "
            "eager execution for scan and FINDING 3's fix is unsound"
        )

    def test_fallback_eager_leg_executes_the_scan_body_once_per_step(self):
        """`r2a_fusion`'s own fallback (for a probe `eager_fn` omits) must
        use the genuinely-eager (per-step) leg, not the compiled-in-disguise
        naive un-jitted call.
        """
        step_n = 4
        calls: list[int] = []

        def transition(carry, x):
            calls.append(1)
            carry = carry + x
            y = {"probe1": carry * 2.0, "final": carry + 1.0}
            return carry, y

        plan = _scan_plan(lane_n=1, step_n=step_n)
        init = jnp.zeros((1,), dtype=jnp.float32)
        xs = jnp.arange(step_n, dtype=jnp.float32)

        def eager_fn(_inputs):
            # Omits BOTH "probe1" and "final" -- like a real caller's
            # model-level oracle that knows nothing about this pipeline's
            # internal probes -- so R2a's fallback covers the whole output
            # in a single un-jitted call (rings.py's `missing_from_eager_fn`
            # branch).
            return {}

        calls.clear()
        result, budgets = rings.r2a_fusion(
            transition,
            plan,
            [jax.ShapeDtypeStruct(xs.shape, xs.dtype)],
            (xs,),
            eager_fn=eager_fn,
            scan_init=init,
        )

        # One trace from `jax.jit(callable_)(*concrete_inputs)` (rings.py's
        # own jit leg) plus, if the fallback is genuinely eager, one Python
        # call per scan step.
        assert len(calls) == 1 + step_n, (
            f"expected 1 (jit trace) + {step_n} (per-step eager fallback) = "
            f"{1 + step_n} transition calls, got {len(calls)} -- the eager "
            f"fallback is compiling the scan body instead of running it "
            f"per-step"
        )
        assert result.passed is True
        assert "probe1" in budgets
        assert "final" in budgets


def _nested_probe_fn(x):
    """A probe (`pair`) whose own value is a nested pytree, not a single array."""
    return {"pair": (x, x * 2.0), "final": x + 1.0}


_NESTED_PROBE_DEPS = {"pair": (), "final": ("pair",)}


class TestDefaultValidateNestedProbe:
    def test_nested_pytree_probe_does_not_refuse_validation(self, toy_plan, toy_abstract_inputs):
        """FINDING 3 (round 3): a nested-pytree-valued probe must not make the
        AC-20 validation gate refuse the WHOLE ladder outright.

        `_probe_result_paths` only resolves a probe whose value is a single
        array (``len(path) <= 1``, its own docstring) -- a probe like
        ``"pair": (a, b)`` has two leaves at path length 2 each, so it gets
        no entry at all. ``validate_probe_deps`` then raises
        ``"probe_result_paths is missing an entry for 'pair' or 'final'"``
        before ANY rung runs, even though R3 itself treats the identical gap
        as best-effort (it catches ``probe_resolution``'s equivalent
        ``ValueError`` and only downgrades to a note).
        """
        # Must not raise -- this is the validation gate's job, and a
        # structural "no single result path" gap is not a violated
        # dependency.
        rings._default_validate(
            _nested_probe_fn, toy_plan, toy_abstract_inputs, None, None, _NESTED_PROBE_DEPS
        )

    def test_the_unvalidated_edge_is_reported_visibly(self, toy_plan, toy_abstract_inputs):
        """The skip must be visible, not silent -- a warning naming the edge."""
        with pytest.warns(UserWarning, match="pair"):
            rings._default_validate(
                _nested_probe_fn, toy_plan, toy_abstract_inputs, None, None, _NESTED_PROBE_DEPS
            )

    def test_a_resolvable_edge_is_still_actually_validated(self, toy_plan, toy_abstract_inputs):
        """The nested-probe skip must not blind validation to a REAL bad edge
        among the resolvable (single-array) probes.
        """

        def fn(x):
            # "final" is declared to depend on "unrelated", but does not --
            # a genuine slice-subset violation among ordinary (non-nested)
            # probes, which must still be caught.
            return {"pair": (x, x * 2.0), "unrelated": x * 0.0, "final": x + 1.0}

        bad_deps = {"pair": (), "unrelated": (), "final": ("pair", "unrelated")}
        with pytest.raises(d.ProbeDependencyError):
            rings._default_validate(fn, toy_plan, toy_abstract_inputs, None, None, bad_deps)


class TestDefaultValidateMissingProbeName:
    """FINDING 2 (round 4): a typo'd/stale ``probe_deps`` name must not fall
    into the same "nested pytree, skip and warn" bucket as a genuine
    nested-pytree probe -- ``_unvalidatable_probe_deps`` used to catch BOTH
    with one check, so a name absent from ``fn``'s output entirely only
    surfaced later, as a bare ``KeyError`` deep inside R3 (~rings.py:1052),
    after R0/R2a/R1/R2b had already compiled and run in full (AC-20: the
    ladder must refuse before any toolchain work when the declared DAG is
    invalid).
    """

    def test_a_name_absent_from_fns_output_raises_before_any_rung(
        self, toy_plan, toy_abstract_inputs
    ):
        # "final" is a typo/stale entry: `_named_fn` only ever returns
        # "probe1" and "final" -- not "finale".
        bad_deps = {"probe1": (), "finale": ("probe1",)}
        with pytest.raises(ValueError, match="finale"):
            rings._default_validate(_named_fn, toy_plan, toy_abstract_inputs, None, None, bad_deps)

    def test_the_raised_message_lists_the_names_that_do_exist(self, toy_plan, toy_abstract_inputs):
        bad_deps = {"probe1": (), "finale": ("probe1",)}
        with pytest.raises(ValueError, match="probe1"):
            rings._default_validate(_named_fn, toy_plan, toy_abstract_inputs, None, None, bad_deps)

    def test_refuses_before_any_toolchain_backed_rung_via_run_ladder(
        self, toy_plan, toy_abstract_inputs
    ):
        """AC-20, end to end: a stale probe name must make `run_ladder`
        refuse before R0/R2a/R1/R2b/R3 ever run, exactly like a genuine
        `ProbeDependencyError` does (`TestRunLadder.
        test_refuses_before_executing_anything_when_validate_rejects`).
        """

        def must_not_run(*_args, **_kwargs):
            raise AssertionError("a rung ran despite an undeclared probe name")

        bad_deps = {"probe1": (), "finale": ("probe1",)}
        with pytest.raises(ValueError, match="finale"):
            rings.run_ladder(
                _named_fn,
                plan=toy_plan,
                abstract_inputs=toy_abstract_inputs,
                concrete_inputs=(),
                probe_deps=bad_deps,
                eager_fn=lambda ci: None,
                r0=must_not_run,
                r1=must_not_run,
                r2a=must_not_run,
                r2b=must_not_run,
                r3=must_not_run,
            )

    def test_bare_output_with_nonempty_probe_deps_raises_with_a_named_top_level_message(
        self, toy_plan, toy_abstract_inputs
    ):
        """`_bare_fn` returns a single array -- no named top level at all.
        Declaring any probe against it (bare or not) must raise a message
        that says probes require named top-level outputs, not the generic
        "hold a nested pytree" wording used for a genuine nested-pytree
        probe.
        """
        bad_deps = {"probe1": ()}
        with pytest.raises(ValueError, match="named top-level"):
            rings._default_validate(_bare_fn, toy_plan, toy_abstract_inputs, None, None, bad_deps)


# ---------------------------------------------------------------------------
# T8b -- run_ladder orchestration (AC-20)
# ---------------------------------------------------------------------------

# SS3.1 (spec revision 5.1), quoted verbatim -- not read off the implementation:
#
#   "Within the non-editing rings, R2a runs first. R2a is the measurement that
#   produces m_leaf, and hence the per-leaf budgets every other float
#   comparison is judged against (SS3.2). Running R1 or R2b ahead of it would
#   leave them with no calibrated budget at all, so the executed sequence is:
#
#   R0 -> R2a -> R1 -> R2b -> R3"
#
# The SS3 ring table lists R1 before R2a because that table is ordered by
# what each ring *varies* (descriptive), not by execution order. Pinning the
# sequence here as a named constant -- rather than inlining the literal list
# at the assertion site -- makes its provenance explicit: a future reader can
# tell the order is spec-pinned, not incidentally whatever the code happens
# to do (AC-20).
SPEC_3_1_EXECUTED_RUNG_ORDER: tuple[str, ...] = ("R0", "R2a", "R1", "R2b", "R3")


class TestRunLadder:
    def test_refuses_before_executing_anything_when_validate_rejects(self):
        """AC-20: refuses before executing anything when validate_probe_deps rejects."""
        calls: list[str] = []

        def fake_validate(*_args, **_kwargs):
            calls.append("validate")
            raise d.ProbeDependencyError("p", "q", frozenset({1}), frozenset())

        def must_not_run(*_args, **_kwargs):
            calls.append("should-not-run")
            raise AssertionError("a rung ran despite validate_probe_deps rejecting the DAG")

        with pytest.raises(d.ProbeDependencyError):
            rings.run_ladder(
                _bare_fn,
                plan=None,
                abstract_inputs=(),
                concrete_inputs=(),
                probe_deps={},
                eager_fn=lambda ci: None,
                validate_fn=fake_validate,
                r0=must_not_run,
                r1=must_not_run,
                r2a=must_not_run,
                r2b=must_not_run,
                r3=must_not_run,
            )
        assert calls == ["validate"]

    def test_runs_rungs_in_section_3_1_order_with_r0_first(self):
        """AC-20: on a well-formed input, runs the rungs in section 3.1's order, R0 first."""
        order: list[str] = []

        def fake_validate(*_args, **_kwargs):
            order.append("validate")

        def make_ring(name):
            def _fn(*_args, **kwargs):
                order.append(name)
                return d.RingResult(
                    ring=name,
                    passed=True,
                    input_class=kwargs["input_class"],
                    in_contract=kwargs["in_contract"],
                    probes=(),
                    notes=(),
                )

            return _fn

        def fake_r2a(fn, plan, ai, ci, **kwargs):  # noqa: ARG001
            order.append("R2a")
            result = d.RingResult(
                ring="R2a",
                passed=True,
                input_class=kwargs["input_class"],
                in_contract=kwargs["in_contract"],
                probes=(),
                notes=(),
            )
            return result, {}

        results = rings.run_ladder(
            _bare_fn,
            plan=None,
            abstract_inputs=(),
            concrete_inputs=(),
            probe_deps={},
            eager_fn=lambda ci: None,
            validate_fn=fake_validate,
            r0=make_ring("R0"),
            r1=make_ring("R1"),
            r2a=fake_r2a,
            r2b=make_ring("R2b"),
            r3=make_ring("R3"),
        )
        assert order == ["validate", *SPEC_3_1_EXECUTED_RUNG_ORDER]
        assert tuple(r.ring for r in results) == SPEC_3_1_EXECUTED_RUNG_ORDER

    def test_r0_gate_failure_skips_the_remaining_rungs_for_that_class(self):
        """AC-16, exercised through run_ladder: no other rung runs once R0 fails."""
        order: list[str] = []

        def fake_validate(*_args, **_kwargs):
            pass

        def fake_r0(fn, plan, ai, ci, **kwargs):  # noqa: ARG001
            order.append("R0")
            return d.RingResult(
                ring="R0",
                passed=False,
                input_class=kwargs["input_class"],
                in_contract=kwargs["in_contract"],
                probes=(),
                notes=("nondeterministic",),
            )

        def must_not_run(*_args, **_kwargs):
            order.append("should-not-run")
            raise AssertionError("a rung ran after R0's hard gate failed")

        results = rings.run_ladder(
            _bare_fn,
            plan=None,
            abstract_inputs=(),
            concrete_inputs=(),
            probe_deps={},
            eager_fn=lambda ci: None,
            validate_fn=fake_validate,
            r0=fake_r0,
            r1=must_not_run,
            r2a=must_not_run,
            r2b=must_not_run,
            r3=must_not_run,
        )
        assert order == ["R0"]
        assert len(results) == 1
        assert results[0].passed is False

    def test_defaults_a_single_nominal_class_when_none_given(self):
        seen_classes: list[str] = []

        def fake_validate(*_args, **_kwargs):
            pass

        def fake_r0(fn, plan, ai, ci, **kwargs):  # noqa: ARG001
            seen_classes.append(kwargs["input_class"])
            return d.RingResult(
                ring="R0",
                passed=False,
                input_class=kwargs["input_class"],
                in_contract=kwargs["in_contract"],
                probes=(),
                notes=(),
            )

        rings.run_ladder(
            _bare_fn,
            plan=None,
            abstract_inputs=(),
            concrete_inputs=(),
            probe_deps={},
            eager_fn=lambda ci: None,
            validate_fn=fake_validate,
            r0=fake_r0,
        )
        assert seen_classes == ["nominal"]

    def test_r0_receives_run_ladders_target(self):
        """FINDING 4: the docstring says target is "used by R0-R3" -- R0's own
        call must receive it, not silently fall back to its NATIVE default
        while R2b/R3 compile a different target.
        """
        seen_targets = []

        def fake_validate(*_args, **_kwargs):
            pass

        def fake_r0(fn, plan, ai, ci, *, target=rings.NATIVE, **kwargs):  # noqa: ARG001
            seen_targets.append(target)
            return d.RingResult(
                ring="R0",
                passed=False,
                input_class=kwargs["input_class"],
                in_contract=kwargs["in_contract"],
                probes=(),
                notes=(),
            )

        rings.run_ladder(
            _bare_fn,
            plan=None,
            abstract_inputs=(),
            concrete_inputs=(),
            probe_deps={},
            eager_fn=lambda ci: None,
            validate_fn=fake_validate,
            r0=fake_r0,
            target=rings.NATIVE_PORTABLE,
        )
        assert seen_targets == [rings.NATIVE_PORTABLE]

    def test_ac17_sub_k_neighbours_label_survives_to_every_report_level_ring_result(self):
        """AC-17: the in-contract label must be asserted at the REPORT level.

        Every ``RingResult``-producing test elsewhere in this suite uses
        ``in_contract=True``, the field's default -- so a runner that dropped
        the label entirely (always defaulting to True) would still pass all
        of them. This drives a real ``sub_k_neighbours``-labelled
        ``InputClassResult`` (out-of-contract, AC-17) through ``run_ladder``
        and checks the ``in_contract`` field on the resulting ``RingResult``s
        themselves -- not just on the generator's own return value, which
        ``TestInputClassGenerators.test_sub_k_neighbours_is_labelled_out_of_contract``
        already covers.
        """
        ic = rings.sub_k_neighbours(64, k_neighbors=48)
        assert ic.label == "sub_k_neighbours"
        assert ic.in_contract is False

        def fake_validate(*_args, **_kwargs):
            pass

        def make_ring(name):
            def _fn(*_args, **kwargs):
                return d.RingResult(
                    ring=name,
                    passed=True,
                    input_class=kwargs["input_class"],
                    in_contract=kwargs["in_contract"],
                    probes=(),
                    notes=(),
                )

            return _fn

        def fake_r2a(fn, plan, ai, ci, **kwargs):  # noqa: ARG001
            result = d.RingResult(
                ring="R2a",
                passed=True,
                input_class=kwargs["input_class"],
                in_contract=kwargs["in_contract"],
                probes=(),
                notes=(),
            )
            return result, {}

        results = rings.run_ladder(
            _bare_fn,
            plan=None,
            abstract_inputs=(),
            concrete_inputs=(),
            probe_deps={},
            input_classes=(ic,),
            eager_fn=lambda ci: None,
            validate_fn=fake_validate,
            r0=make_ring("R0"),
            r1=make_ring("R1"),
            r2a=fake_r2a,
            r2b=make_ring("R2b"),
            r3=make_ring("R3"),
        )
        assert results  # sanity: the ladder actually ran
        for result in results:
            assert result.input_class == "sub_k_neighbours"
            assert result.in_contract is False


# ---------------------------------------------------------------------------
# Toolchain-gated real end-to-end smoke test
# ---------------------------------------------------------------------------


class TestRealToolchainSmoke:
    """Same importorskip pattern as test_compile_native_wasm32.py / test_parity_multi_size.py."""

    def test_r0_and_r2a_against_a_real_toolchain(
        self, model, plan, xs, abstract_inputs, reference_fn
    ):
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        def fn(x):
            return model(x)

        r0_result = rings.r0_replay_gate(fn, plan, abstract_inputs, (xs,), replays=2)
        assert r0_result.ring == "R0"
        assert r0_result.passed is True

        r2a_result, budgets = rings.r2a_fusion(
            fn, plan, abstract_inputs, (xs,), eager_fn=reference_fn
        )
        assert r2a_result.ring == "R2a"
        assert r2a_result.passed is True
        assert budgets  # at least one float leaf measured
