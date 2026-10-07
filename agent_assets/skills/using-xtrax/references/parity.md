# Reference parity

Compare a candidate sampler to an independent oracle with `xtrax.testing`
(`src/xtrax/testing/`). The package takes callables and NumPy arrays. A torch
reference stays in the caller.

Signatures below match the source (`inspect.signature` on this tree).

## Seeds are a different stream on each side

JAX and torch do not share a generator. The same integer seed draws different
uniforms, different normals, and a different permutation. Match the two sides
by handing both one host array.

## Strategy order

Use the first strategy the sampler allows.

1. **Inject the randomness.** Build one `InjectedSource` and `bind` it to both
   callables. `order_from_randn(mask, randn, eps)` is the shared decoding order:
   a stable ascending argsort of `(mask + eps) * abs(randn)`. Equal scores keep
   the lower index, matching `numpy.argsort(..., kind="stable")`,
   `jax.numpy.argsort`, and `torch.argsort(..., stable=True)`. Pass that index
   array into both models.
2. **Teacher-force both directions.** `teacher_forced_lane` scores one fixed
   sequence along `order` and along `order[::-1]`.
3. **Collapse the sampler.** `collapsed_sampler_lane` compares temperature-0
   argmax. A site is a near-tie when the smaller of the candidate and oracle
   top-two margins is at or below `tie_margin`. That site is reported and is
   not a failure.
4. **Compare one-step conditionals.** `distributional_lane` tests N draws
   against oracle probabilities with a chi-square tail (`jax.scipy.stats.chi2.sf`)
   and total variation, and reports `min_detectable_tv` for that N and alpha
   (default power 0.8, least-favorable alternative `TV = 0.5 * sqrt(λ / n)`).
   Across T steps, rejection uses Holm's step-down procedure at family-wise
   level `alpha`: ordered p-values are compared with `alpha / (T - rank)`, and
   the first p-value that is not below its threshold stops the procedure.

## Call the public entry point

The callable under test is the one a user calls: the run spec, the runner, or
the CLI. Bind that entry point. A match on the model forward alone leaves the
dispatch path unchecked.

## A second implementation

`assert_distinct_callables(candidate, oracle)` raises `SelfParityError` when the
two arguments are the same object, or the same bound method. Running one
function twice measures determinism. Parity is a comparison against another
implementation.

## Negative controls

Each lane's evidence includes a case built to fail: a swapped alphabet on the
teacher-forced lane, a decisive argmax miss on the collapsed lane, a biased
sampler on the distributional lane. `require_negative_control` perturbs a named
knob and requires the chi-square test to reject. It is the only constructor of
`NegativeControl`; the control records the knob, the perturbed run's draw count
`n`, and `alpha`. A hand-built `NegativeControl` raises `TypeError`.
`distributional_lane` reports `PASS` only when that control rejected, its `n`
and `alpha` match the lane, and the unperturbed draws still match under Holm.
A control whose `n` or `alpha` differs raises `ValueError`. With no rejecting
control the verdict is `UNCONTROLLED`.

`knob_coverage(declared, varied)` lists declared knobs that were not varied, so
an unexercised knob stays visible in the evidence.

## Signatures

```python
def order_from_randn(mask: ArrayLike, randn: ArrayLike, eps: float) -> np.ndarray: ...

class InjectedSource:
    order: np.ndarray | None = None
    noise: np.ndarray | None = None
    uniform: np.ndarray | None = None
    def parameters(self) -> dict[str, np.ndarray]: ...
    def bind(self, fn: Callable[..., Any]) -> Callable[..., Any]: ...

def assert_distinct_callables(
    candidate: Callable[..., Any], oracle: Callable[..., Any]
) -> None: ...

# Scorer: (tokens, position, known) -> logits of shape (V,)
Scorer = Callable[[np.ndarray, int, np.ndarray], np.ndarray]

def align_logits(logits: ArrayLike, index_map: ArrayLike) -> np.ndarray: ...
def map_token_ids(tokens: ArrayLike, index_map: ArrayLike) -> np.ndarray: ...

def teacher_forced_lane(
    sequence: ArrayLike,
    order: ArrayLike,
    candidate: Scorer,
    oracle: Scorer,
    *,
    alphabet_map: ArrayLike | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TeacherForcedResult: ...

def collapsed_sampler_lane(
    candidate_logits: ArrayLike,
    oracle_logits: ArrayLike,
    *,
    tie_margin: float,
    alphabet_map: ArrayLike | None = None,
) -> CollapsedSamplerResult: ...

def min_detectable_tv(
    n: int,
    n_classes: int,
    alpha: float = 0.05,
    *,
    power: float = 0.8,
) -> float: ...

def distributional_lane(
    draws: ArrayLike,
    oracle_probs: ArrayLike,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    negative_control: NegativeControl | None = None,
) -> DistributionalResult: ...

def require_negative_control(
    sampler: Callable[[Mapping[str, Any]], np.ndarray],
    oracle_probs: ArrayLike,
    knobs: Mapping[str, Any],
    knob: str,
    perturbed_value: Any,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> NegativeControl: ...

def knob_coverage(declared: Sequence[str], varied: Sequence[str]) -> KnobCoverage: ...
```

`alphabet_map[i]` is the oracle class for candidate class `i`. Token ids are
mapped before the oracle scorer runs; oracle logits are gathered back with
`align_logits`. Draws for `distributional_lane` are integer ids of shape `(N,)`
or `(N, T)`, with oracle probabilities `(K,)` or `(T, K)`.
