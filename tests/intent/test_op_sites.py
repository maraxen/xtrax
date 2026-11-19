"""Tests for ``xtrax.op`` calls as intent sites of kind ``"op"`` (backlog #6098, step 2).

Kernels are Pallas in interpret mode (CPU), shared with ``tests/op``. A Context with
``kernels=True`` and ``backend="cpu"`` selects them. Every "the kernel ran" claim is read from
the jaxpr (``pallas_call``), never from output equality.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.op._kernels import (
    make_hist_kernel,
    make_hist_reference,
    matmul_kernel,
    matmul_reference,
    scale_kernel,
    scale_reference,
)
from xtrax.intent import (
    Context,
    IntentError,
    OpPrimal,
    Plan,
    SiteDecision,
    SiteKey,
    SiteMismatchError,
    compile,
    loop,
    map,
)
from xtrax.intent.sites import Session, activate
from xtrax.kernel_seam import VjpKernel, op
from xtrax.testing.kernel_parity import KernelParityError, assert_kernel_ran
from xtrax.tiling.strategy import Vmap

CPU_KERNELS = Context(kernels=True, backend="cpu")
CPU_REFERENCE = Context(kernels=False, backend="cpu")
EXPORT = Context(target="export", export_format="stablehlo", kernels=True, backend="cpu")
NO_TPU_KERNEL = Context(kernels=True, backend="tpu")


def _mm(**kwargs):
    return op(matmul_reference, kernels={"cpu": matmul_kernel}, name="mm", **kwargs)


def _mm_vjp_op():
    def fwd(x, w):
        return matmul_kernel(x, w), (x, w)

    def bwd(res, ct):
        x, w = res
        return matmul_kernel(ct, w.T), matmul_kernel(x.T, ct)

    return op(matmul_reference, vjp_kernels={"cpu": VjpKernel(fwd, bwd)}, name="mm_vjp")


def _args():
    kx, kw = jax.random.split(jax.random.PRNGKey(0))
    return jax.random.normal(kx, (16, 24)), jax.random.normal(kw, (24, 16))


def _mixed(mm):
    """A loop, a map and an op in one function."""

    def fn(xs, w):
        def body(carry, x):
            return carry + x, jnp.sum(carry)

        carry, ys = loop(body, jnp.zeros(xs.shape[1:], xs.dtype), xs=xs, name="steps")
        scaled = map(lambda row: row * 2.0, xs, name="rows")
        return mm(scaled, w) + carry.sum() + ys.sum()

    return fn


def _reference_mixed():
    def fn(xs, w):
        carry, ys = jax.lax.scan(lambda c, x: (c + x, jnp.sum(c)), jnp.zeros(xs.shape[1:]), xs)
        return (xs * 2.0) @ w + carry.sum() + ys.sum()

    return fn


def _jaxpr(compiled, *args) -> str:
    return str(jax.make_jaxpr(lambda *a: compiled(*a))(*args))


def _has_pallas(compiled, *args) -> bool:
    return "pallas_call" in _jaxpr(compiled, *args)


def _op_rows(plan: Plan) -> list[SiteDecision]:
    return [d for d in plan.decisions if d.kind == "op"]


# --- the plan names the op ---------------------------------------------------------------------


def test_mixed_function_lists_the_op_row():
    xs, w = _args()
    plan = compile(_mixed(_mm()), CPU_KERNELS).plan_for(xs, w)
    kinds = [d.kind for d in plan.decisions]
    assert kinds.count("op") == 1 and "loop" in kinds and "map" in kinds
    (row,) = _op_rows(plan)
    assert row.key == SiteKey(parent=None, path="primal", label="mm", ordinal=0)
    assert row.primal == OpPrimal(backend="cpu", form="custom_jvp")
    assert row.primal.label == "kernel:cpu"
    assert row.primal_reason == "kernel for backend 'cpu'"
    assert row.diff is None and "reference" in row.diff_reason
    table = plan.table()
    assert "primal/mm#0\top\tkernel:cpu (custom_jvp)\tNone\tkernel for backend 'cpu'" in table
    # The loop and the map rows still print.
    assert "primal/steps#0\tloop" in table and "primal/rows#0\tmap" in table


@pytest.mark.parametrize(
    ("ctx", "reason"),
    [
        (CPU_REFERENCE, "kernels=False"),
        (EXPORT, "export target"),
        (NO_TPU_KERNEL, "no kernel for backend 'tpu'"),
    ],
)
def test_reference_row_carries_its_reason(ctx, reason):
    xs, w = _args()
    plan = compile(_mixed(_mm()), ctx).plan_for(xs, w)
    (row,) = _op_rows(plan)
    assert row.primal == OpPrimal(backend=None, form="reference")
    assert row.primal_reason == reason
    assert f"primal/mm#0\top\treference\tNone\t{reason}" in plan.table()


# --- the jaxpr agrees with the plan -------------------------------------------------------------


@pytest.mark.parametrize(
    "ctx",
    [CPU_KERNELS, CPU_REFERENCE, EXPORT, NO_TPU_KERNEL],
    ids=["kernel", "kernels-off", "export", "no-kernel-for-backend"],
)
def test_pallas_call_in_jaxpr_exactly_when_the_plan_says_kernel(ctx):
    xs, w = _args()
    compiled = compile(_mixed(_mm()), ctx)
    plan = compiled.plan_for(xs, w)
    (row,) = _op_rows(plan)
    planned_kernel = row.primal.backend is not None
    assert planned_kernel == (ctx == CPU_KERNELS)
    assert _has_pallas(compiled, xs, w) == planned_kernel
    np.testing.assert_allclose(compiled(xs, w), _reference_mixed()(xs, w), rtol=1e-4, atol=1e-4)


def test_jaxpr_probe_goes_red_on_a_reference_run():
    """Negative control: the same probe the plan test relies on fails when the reference ran."""
    xs, w = _args()
    ran = compile(_mixed(_mm()), CPU_KERNELS)
    fell_back = compile(_mixed(_mm()), CPU_REFERENCE)
    assert_kernel_ran(lambda a, b: ran(a, b), xs, w)
    with pytest.raises(KernelParityError):
        assert_kernel_ran(lambda a, b: fell_back(a, b), xs, w)


def test_export_target_has_no_pallas_call_and_exports():
    xs, w = _args()
    compiled = compile(_mixed(_mm()), EXPORT)
    assert not _has_pallas(compiled, xs, w)
    exported = jax.export.export(jax.jit(lambda a, b: compiled(a, b)))(xs, w)
    np.testing.assert_allclose(
        exported.call(xs, w), _reference_mixed()(xs, w), rtol=1e-4, atol=1e-4
    )


# --- derivatives ---------------------------------------------------------------------------------


def test_grad_through_a_compiled_op_site_matches_the_reference():
    xs, w = _args()
    compiled = compile(_mixed(_mm()), CPU_KERNELS)
    got = jax.grad(lambda a, b: compiled(a, b).sum(), argnums=(0, 1))(xs, w)
    want = jax.grad(lambda a, b: _reference_mixed()(a, b).sum(), argnums=(0, 1))(xs, w)
    for g, r in zip(got, want, strict=True):
        np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-4)
    # The kernel is still the primal while differentiating.
    assert "pallas_call" in str(jax.make_jaxpr(jax.grad(lambda a: compiled(a, w).sum()))(xs))


def test_grad_comparison_is_sensitive_to_a_wrong_derivative():
    """Negative control: a reference with a different derivative makes the comparison fail."""
    xs, w = _args()
    wrong = op(lambda x, y: 1.01 * (x @ y), kernels={"cpu": matmul_kernel}, name="mm")
    compiled = compile(_mixed(wrong), CPU_KERNELS)
    got = jax.grad(lambda a, b: compiled(a, b).sum(), argnums=1)(xs, w)
    want = jax.grad(lambda a, b: _reference_mixed()(a, b).sum(), argnums=1)(xs, w)
    assert not np.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_vjp_kernel_form_is_planned_and_reverse_mode_matches():
    xs, w = _args()
    compiled = compile(_mixed(_mm_vjp_op()), CPU_KERNELS)
    plan = compiled.plan_for(xs, w)
    (row,) = _op_rows(plan)
    assert row.primal == OpPrimal(backend="cpu", form="custom_vjp")
    assert "kernel:cpu (custom_vjp)" in plan.table()
    assert "forward mode" in row.diff_reason
    got = jax.grad(lambda a, b: compiled(a, b).sum(), argnums=(0, 1))(xs, w)
    want = jax.grad(lambda a, b: _reference_mixed()(a, b).sum(), argnums=(0, 1))(xs, w)
    for g, r in zip(got, want, strict=True):
        np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-4)


def test_integer_output_op_site_is_planned_and_usable_under_grad():
    nbins = 5
    hist = op(
        make_hist_reference(nbins),
        kernels={"cpu": make_hist_kernel(nbins)},
        name="hist",
    )

    def fn(x, idx):
        counts = hist(idx)
        return jnp.sum(x * counts[:3].astype(x.dtype))

    x = jnp.arange(3.0)
    idx = jnp.array([0, 1, 1, 2, 4, 4, 4, 0], dtype=jnp.int32)
    compiled = compile(fn, CPU_KERNELS)
    (row,) = _op_rows(compiled.plan_for(x, idx))
    assert row.primal.label == "kernel:cpu"
    np.testing.assert_allclose(jax.grad(compiled)(x, idx), np.array([2.0, 2.0, 1.0]))
    assert "pallas_call" in str(jax.make_jaxpr(jax.grad(compiled))(x, idx))


# --- Plan JSON ----------------------------------------------------------------------------------


def test_plan_json_round_trips_op_sites():
    xs, w = _args()
    for ctx in (CPU_KERNELS, CPU_REFERENCE):
        plan = compile(_mixed(_mm()), ctx).plan_for(xs, w)
        restored = Plan.from_json(plan.to_json(), context=ctx)
        assert restored == plan
        assert restored.table() == plan.table()
    payload = json.loads(compile(_mixed(_mm()), CPU_KERNELS).plan_for(xs, w).to_json())
    (row,) = [d for d in payload["decisions"] if d["kind"] == "op"]
    assert row["primal"] == {"type": "OpPrimal", "backend": "cpu", "form": "custom_jvp"}


def test_plan_json_without_op_sites_is_unchanged_and_old_plans_load():
    xs, w = _args()
    plan = compile(_reference_mixed_intent(), CPU_KERNELS).plan_for(xs, w)
    assert not _op_rows(plan)
    assert "OpPrimal" not in plan.to_json()
    assert Plan.from_json(plan.to_json(), context=CPU_KERNELS) == plan
    # A plan written before op sites existed: a map site, no `warnings` key at plan level.
    old = json.dumps(
        {
            "schema_version": 1,
            "context": "Context()",
            "decisions": [
                {
                    "key": {"parent": None, "path": "primal", "label": "rows", "ordinal": 0},
                    "kind": "map",
                    "primal": {"type": "Vmap"},
                    "diff": {"type": "Vmap"},
                    "primal_reason": "no memory budget; cardinality selects Vmap",
                    "diff_reason": "same strategy as primal in the MVP",
                    "warnings": [],
                }
            ],
        }
    )
    loaded = Plan.from_json(old, context=Context())
    assert loaded.decisions[0].kind == "map" and loaded.decisions[0].primal == Vmap()


def _reference_mixed_intent():
    def fn(xs, w):
        scaled = map(lambda row: row * 2.0, xs, name="rows")
        return scaled @ w

    return fn


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "OpPrimal", "backend": "cpu", "form": "custom_jacobian"},
        {"type": "OpPrimal", "backend": None, "form": "custom_jvp"},
        {"type": "OpPrimal", "backend": "cpu", "form": "reference"},
        {"type": "OpPrimal", "backend": "cpu"},
    ],
)
def test_plan_json_rejects_a_malformed_op_primal(bad):
    xs, w = _args()
    payload = json.loads(compile(_mixed(_mm()), CPU_KERNELS).plan_for(xs, w).to_json())
    for decision in payload["decisions"]:
        if decision["kind"] == "op":
            decision["primal"] = bad
    with pytest.raises(ValueError, match="OpPrimal|form|backend"):
        Plan.from_json(json.dumps(payload), context=CPU_KERNELS)


# --- one function, two contexts ------------------------------------------------------------------


def test_one_function_compiled_under_two_contexts_never_crosses_decisions():
    xs, w = _args()
    fn = _mixed(_mm())
    kernel = compile(fn, CPU_KERNELS)
    reference = compile(fn, CPU_REFERENCE)
    # Interleave: plan and run each twice so a cache keyed by ``fn`` alone would cross.
    for _ in range(2):
        assert _has_pallas(kernel, xs, w)
        assert not _has_pallas(reference, xs, w)
    (kernel_row,) = _op_rows(kernel.plan_for(xs, w))
    (reference_row,) = _op_rows(reference.plan_for(xs, w))
    assert kernel_row.primal.label == "kernel:cpu"
    assert reference_row.primal.label == "reference"
    np.testing.assert_allclose(kernel(xs, w), reference(xs, w), rtol=1e-4, atol=1e-4)


# --- explicit context: the compile context wins over a bound one ---------------------------------


def test_compile_context_overrides_a_bound_op_and_the_plan_says_so():
    xs, w = _args()
    bound_kernel = _mm().with_context(CPU_KERNELS)
    compiled = compile(_mixed(bound_kernel), CPU_REFERENCE)
    (row,) = _op_rows(compiled.plan_for(xs, w))
    assert row.primal.label == "reference" and row.primal_reason == "kernels=False"
    assert len(row.warnings) == 1
    assert "with_context" in row.warnings[0] and "kernel:cpu" in row.warnings[0]
    assert "compile context wins" in row.warnings[0]
    assert not _has_pallas(compiled, xs, w)
    # Outside compile the binding still applies.
    assert bound_kernel.select().primal == "kernel:cpu"


def test_compile_context_can_turn_a_reference_binding_into_a_kernel():
    xs, w = _args()
    bound_reference = _mm().with_context(CPU_REFERENCE)
    compiled = compile(_mixed(bound_reference), CPU_KERNELS)
    (row,) = _op_rows(compiled.plan_for(xs, w))
    assert row.primal.label == "kernel:cpu" and "reference" in row.warnings[0]
    assert _has_pallas(compiled, xs, w)


def test_no_warning_when_the_binding_makes_no_difference_or_is_absent():
    xs, w = _args()
    for mm in (_mm(), _mm().with_context(CPU_KERNELS), _mm().with_context(CPU_REFERENCE)):
        ctx = CPU_REFERENCE if mm.context == CPU_REFERENCE or not mm.bound else CPU_KERNELS
        (row,) = _op_rows(compile(_mixed(mm), ctx).plan_for(xs, w))
        assert row.warnings == ()


# --- sites inside loops and maps ------------------------------------------------------------------


def test_op_inside_a_loop_body_is_a_child_site_on_both_paths():
    scale = op(scale_reference, kernels={"cpu": scale_kernel}, name="scale")

    def fn(xs):
        def body(carry, x):
            return carry + scale(x), jnp.sum(x)

        carry, ys = loop(body, jnp.zeros(xs.shape[1:], xs.dtype), xs=xs, name="steps")
        return carry.sum() + ys.sum()

    xs = jnp.arange(48.0).reshape(6, 8)
    compiled = compile(fn, CPU_KERNELS)
    plan = compiled.plan_for(xs)
    ops = _op_rows(plan)
    paths = {str(row.key) for row in ops}
    assert "primal/steps#0/primal/scale#0" in paths
    assert any("/diff/scale#" in p for p in paths)
    assert all(row.primal.label == "kernel:cpu" for row in ops)
    assert _has_pallas(compiled, xs)
    np.testing.assert_allclose(jax.grad(compiled)(xs), jnp.full_like(xs, 4.0))
    assert not _has_pallas(compile(fn, CPU_REFERENCE), xs)


def test_op_inside_a_map_body_is_a_child_site():
    scale = op(scale_reference, kernels={"cpu": scale_kernel}, name="scale")

    def fn(xs):
        return map(lambda row: scale(row), xs, name="rows").sum()

    xs = jnp.arange(48.0).reshape(6, 8)
    compiled = compile(fn, CPU_KERNELS)
    ops = _op_rows(compiled.plan_for(xs))
    assert any(str(row.key).startswith("primal/rows#0/primal/scale#0") for row in ops)
    assert _has_pallas(compiled, xs)
    np.testing.assert_allclose(compiled(xs), 3.0 * xs.sum())


def test_sites_inside_a_selected_reference_are_children_of_the_op():
    def reference(xs):
        carry, _ = loop(lambda c, x: (c + x, c), jnp.zeros(xs.shape[1:]), xs=xs, name="inner")
        return carry

    inner = op(reference, name="outer_op")
    xs = jnp.arange(24.0).reshape(3, 8)
    plan = compile(lambda a: inner(a), CPU_REFERENCE).plan_for(xs)
    labels = [str(d.key) for d in plan.decisions]
    assert "primal/outer_op#0" in labels
    assert "primal/outer_op#0/primal/inner#0" in labels


# --- explicit session context ------------------------------------------------------------------


def test_session_context_selects_not_ambient_state():
    xs, w = _args()
    mm = _mm()
    with activate(Session("plan", context=CPU_KERNELS)) as session:
        jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w)
    (record,) = session.records
    assert record.kind == "op" and record.op.primal.label == "kernel:cpu"
    with activate(Session("plan", context=CPU_REFERENCE)) as session:
        jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w)
    assert session.records[0].op.primal.label == "reference"


def test_session_without_a_context_uses_the_ops_bound_context():
    xs, w = _args()
    bound = _mm().with_context(CPU_KERNELS)
    with activate(Session("plan")) as session:
        text = str(jax.make_jaxpr(lambda a, b: bound(a, b))(xs, w))
    assert "pallas_call" in text
    assert session.records[0].op.primal.label == "kernel:cpu"


def test_exec_follows_the_planned_decision_and_rejects_a_kernel_the_op_lacks():
    xs, w = _args()
    mm = _mm()
    with activate(Session("plan", context=CPU_KERNELS)) as planned:
        jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w)
    (record,) = planned.records
    key = record.key

    def decision(primal):
        return SiteDecision(key, "op", primal, None, "planned", "n/a")

    # The decision, not the context, picks the implementation in exec mode.
    reference = decision(OpPrimal(backend=None, form="reference"))
    with activate(Session("exec", planned.records, {key: reference}, context=CPU_KERNELS)):
        assert "pallas_call" not in str(jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w))
    kernel = decision(OpPrimal(backend="cpu", form="custom_jvp"))
    with activate(Session("exec", planned.records, {key: kernel}, context=CPU_REFERENCE)):
        assert "pallas_call" in str(jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w))
    # A plan that names a kernel this op does not have is an error, not a silent fallback.
    for primal in (OpPrimal(backend="gpu", form="custom_jvp"), OpPrimal("cpu", "custom_vjp")):
        planned_with = {key: decision(primal)}
        with activate(Session("exec", planned.records, planned_with, context=CPU_KERNELS)):
            with pytest.raises(IntentError, match="no .*kernel"):
                jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w)
    # A strategy that is not an op primal is an error too.
    with activate(Session("exec", planned.records, {key: decision(Vmap())}, context=CPU_KERNELS)):
        with pytest.raises(IntentError, match="OpPrimal"):
            jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w)


def test_exec_session_rejects_an_op_call_whose_shapes_differ_from_the_plan():
    xs, w = _args()
    mm = _mm()
    with activate(Session("plan", context=CPU_KERNELS)) as planned:
        jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w)
    decisions = {
        planned.records[0].key: SiteDecision(
            planned.records[0].key, "op", OpPrimal("cpu", "custom_jvp"), None, "planned", "n/a"
        )
    }
    with activate(Session("exec", planned.records, decisions, context=CPU_KERNELS)):
        with pytest.raises(SiteMismatchError):
            jax.make_jaxpr(lambda a, b: mm(a, b))(xs, w[:, :8])


def test_override_on_an_op_label_is_refused_not_misapplied():
    xs, w = _args()
    with pytest.raises(IntentError, match="op"):
        compile(_mixed(_mm()), CPU_KERNELS, overrides={"mm": (Vmap(), None)}).plan_for(xs, w)


def test_sites_in_the_reference_stay_out_of_the_plan_while_a_kernel_runs():
    def reference(xs):
        carry, _ = loop(lambda c, x: (c + x, c), jnp.zeros(xs.shape[1:]), xs=xs, name="inner")
        return carry

    # A plain-JAX stand-in for a kernel: the point is the derivative, which runs the reference.
    fast = op(reference, kernels={"cpu": lambda xs: xs.sum(0)}, name="fast")
    xs = jnp.arange(24.0).reshape(3, 8)
    compiled = compile(lambda a: fast(a).sum(), CPU_KERNELS)
    assert [str(d.key) for d in compiled.plan_for(xs).decisions] == ["primal/fast#0"]
    np.testing.assert_allclose(jax.grad(compiled)(xs), jnp.ones_like(xs))
