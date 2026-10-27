> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# xtrax.op

`xtrax.op(reference, kernels=..., vjp_kernels=...)` puts a plain-JAX reference and per-backend kernels (Pallas, or any function with the reference's signature) behind one callable. The caller's `Context` decides which one runs. It replaces a hand-rolled `use_pallas` flag, environment variable or `interpret=` switch. The reference defines the output, the derivatives and the export path. — verify: `src/xtrax/kernel_seam.py:1-25` — verify: `src/xtrax/kernel_seam.py:585-630`

`op`, `Op` and `VjpKernel` are exported from `xtrax`. — verify: `src/xtrax/__init__.py:89` — verify: `src/xtrax/__init__.py:100` — verify: `src/xtrax/__init__.py:126`

## Example

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import xtrax
from xtrax.intent import Context

def scale_reference(x, *, n):
    return x * n

def _body(x_ref, o_ref, *, n):
    o_ref[...] = x_ref[...] * n

def scale_kernel(x, *, n, interpret):
    return pl.pallas_call(
        functools.partial(_body, n=n),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)

def make_scale_op(n: int, ctx: Context) -> xtrax.Op:
    return xtrax.op(
        functools.partial(scale_reference, n=n),
        kernels={
            "gpu": functools.partial(scale_kernel, n=n, interpret=False),
            "cpu": functools.partial(scale_kernel, n=n, interpret=True),
        },
        name="scale",
        context=ctx,
    )

scale = make_scale_op(3, Context(kernels=True))
y = jax.jit(scale)(jnp.ones((8,), jnp.float32))
g = jax.grad(lambda x: scale(x).sum())(jnp.ones((8,), jnp.float32))  # derivative from the reference
```

## Selection

`Op.select(ctx)` applies these rules in order. — verify: `src/xtrax/kernel_seam.py:312-338`

1. `ctx.target == "export"` selects the reference (`"export target"`).
2. `ctx.kernels` is `False` selects the reference (`"kernels=False"`). This is the default `Context()`, so an op built without `context=` runs the reference. — verify: `src/xtrax/intent/context.py:30-35` — verify: `src/xtrax/kernel_seam.py:265-266`
3. A `vjp_kernels` entry for `ctx.resolved_backend()` selects the `custom_vjp` form.
4. A `kernels` entry for that backend selects the `custom_jvp` form.
5. Otherwise the reference runs (`"no kernel for backend '<b>'"`).

`resolved_backend()` is `ctx.backend`, or `jax.default_backend()` when that is unset. Kernel keys are therefore backend names (`"gpu"`, `"tpu"`, `"cpu"`). — verify: `src/xtrax/intent/context.py:60-73`

`Context` has no interpret flag. To run a Pallas kernel on CPU, register a `"cpu"` entry built with `interpret=True`. Without one, CPU runs the reference. — verify: `src/xtrax/intent/context.py:30-35`

An op is immutable. `with_context(ctx)` returns a new op, so two differently bound ops are different callables, and a `jax.jit` cache entry made for one is never reused for the other. — verify: `src/xtrax/kernel_seam.py:227-231` — verify: `src/xtrax/kernel_seam.py:296-310`

Inside `xtrax.intent.compile`, the compile's context wins over a bound one, and the plan records the choice as a site of kind `"op"`: read `references/intent.md`. — verify: `src/xtrax/kernel_seam.py:370-378`

## Arguments are traced: close over static configuration

An op is called with positional arguments only, and every one of them is traced: the kernel form is a `jax.custom_jvp` over all of them. Grid sizes, spline orders, tile shapes and the `interpret` flag go into the reference and the kernels by closure (`functools.partial`, or a factory such as `make_scale_op` above), never as call arguments. — verify: `src/xtrax/kernel_seam.py:340-342` — verify: `src/xtrax/kernel_seam.py:504-511`

Passing a Python `int` as an op argument fails in eager, `jit` and `grad` alike. The int arrives as a traced `i32[]`, and Pallas rejects it with `The kernel function in the pallas_call body ... captures constants [i32[]]. You should pass them as inputs.` The same op with the int closed over by `functools.partial` runs, differentiates, and shows `pallas_call` in its jaxpr. (Observed 261027, jax 0.11.1, CPU interpret mode.)

## Output check

At call time a selected kernel's output structure, shapes and dtypes are compared with the reference's via `jax.eval_shape`, and a mismatch raises `IntentError`. The check runs at trace time only. — verify: `src/xtrax/kernel_seam.py:486-502`

## Derivatives

- **`kernels` (the `custom_jvp` form).** The primal is the kernel. The JVP is `jax.jvp(reference)`, with `float0` tangents for integer and bool inputs. Reverse mode, `vmap` and Hessians follow, because JAX transposes that linear JVP. A kernel that uses `atomic_add` or another non-differentiable primitive is never differentiated: only the reference is. — verify: `src/xtrax/kernel_seam.py:504-522` — verify: `src/xtrax/kernel_seam.py:11-16`
- **`vjp_kernels` (the `custom_vjp` form).** Pass a `VjpKernel(fwd, bwd)`, where `fwd(*args) -> (out, residuals)` and `bwd(residuals, ct) -> one cotangent per argument`. Reverse mode runs `bwd`. Second-order reverse mode falls back to the reference. Forward mode (`jvp`, `jacfwd`) raises `IntentError`. A backend present in both mappings uses the vjp kernel. — verify: `src/xtrax/kernel_seam.py:66-82` — verify: `src/xtrax/kernel_seam.py:524-582` — verify: `src/xtrax/kernel_seam.py:448-466`
- **Nested ops.** While a derivative rule differentiates a reference, every op inside that reference runs its own reference. Forward passes still run kernels. `jax.jit` caches do not see this switch, so do not wrap an inner op in its own `jax.jit` inside a reference. — verify: `src/xtrax/kernel_seam.py:27-31` — verify: `src/xtrax/kernel_seam.py:370-371`

## Parity tests

`xtrax.testing.kernel_parity` holds the helpers. Float leaves use the scale-relative tolerance `rel * max(1, max|ref|)`; integer and bool leaves need exact equality. An elementwise `1e-5` fails at `|ref| ~ 150` from float32 accumulation order. — verify: `src/xtrax/testing/kernel_parity.py:1-13`

A parity test has three parts:

1. **Parity.** `assert_kernel_parity(kernel, reference, *args, rel=1e-5)`. — verify: `src/xtrax/testing/kernel_parity.py:163-192`
2. **The kernel ran.** `assert_kernel_ran(op_under_ctx, *args)` looks for a `pallas_call` in the jaxpr. A kernel whose output equals the reference is often a silent fallback to the reference. — verify: `src/xtrax/testing/kernel_parity.py:195-221`
3. **A negative control.** `require_corruption_red(kernel, reference, args, corrupt)` must see a deliberately broken kernel fail. A parity check that cannot fail proves nothing. — verify: `src/xtrax/testing/kernel_parity.py:224-261`

For a backward kernel, use `assert_vjp_parity(vk, reference, args, key=...)`, which applies a random cotangent. An all-ones cotangent hides sign and permutation errors. Pair it with `require_vjp_controls_red(vk, reference, args, weight_argnum=..., key=...)`, which runs the 0.1% scale, permuted-weight and dropped-gradient controls. — verify: `src/xtrax/testing/kernel_parity.py:282-345` — verify: `src/xtrax/testing/kernel_parity.py:461-480`

The xtrax tests themselves follow this pattern, with Pallas in interpret mode. — verify: `tests/op/test_kernel_parity.py:83-180` — verify: `tests/op/_kernels.py:1-50`

## Interpret mode is not GPU evidence

CPU parity in interpret mode tests the kernel's math, not its GPU lowering. Two hazards were measured on jax 0.11.1 on 261026:

- **Pallas Triton `atomic_add` writes nothing on sm120.** On an RTX PRO 6000 Blackwell, every `atomic_add` variant into an output block left the output all zero, while plain stores worked. Interpret mode is exact, so CPU tests pass. Before trusting an atomic kernel (a scatter, a histogram, charge spreading), run a GPU totals check, such as row sums equal to the count. The mechanism is unverified, and Hopper (H200) has not been checked.
- **Pre-Hopper GPUs run no compiled Pallas.** `pallas_call` lowers to Mosaic GPU by default, and on pre-Hopper hardware it supports only input-only pipelines. Triton (`compiler_params=pallas.triton.CompilerParams()`) is deprecated in 0.11.1 and accepts a fixed list of device kinds that excludes Turing. Run compiled kernels on H200-class GPUs.

## Not built

- An op cannot declare supported dtypes or determinism, so there is no "dtype not supported" or "non-deterministic kernel refused" selection reason. — verify: `src/xtrax/kernel_seam.py:60-63`
- An override cannot name an op's implementation inside `compile`: read `references/intent.md`.
