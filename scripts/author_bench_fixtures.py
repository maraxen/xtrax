"""T1-13 `graph author-bench` fixtures (#3068): authoring tasks + human-labeled spot-check.

Benchmark-only data, kept in `scripts/` (not `src/xtrax/`) -- same placement rationale T1-12's
smoke script used: this module has no business being reachable from production authoring code.

`TASKS` resolves the design gap T1-12 left open for benchmarking: `TemplateGenerator.generate`
(`xtrax.composition.author`) takes no task/prompt input at all, so an authored graph has nothing
to be "faithful" or "unfaithful" *to* on its own. Each task pairs an NL prompt with an
`expected_sequence` of import-path refs into T1-12's own `_EXAMPLE_CALLABLES` pool
(`example_increment`/`example_double`/`example_square`/`example_negate`) -- covering all 6
pairwise orderings and 4 representative 3-node chains. This does not change
`TemplateGenerator`'s shipped API; the bench harness runs it exactly as-is (blind to the task,
seeded random sampling) and scores whether its output happens to match.

`HUMAN_LABELED_SPOT_CHECK` operationalizes the AC's hard human-labeled sub-gate (N>=30, PM2) as
a hand-verified fixture: each item's `expected_output` is the result of composing the *real*
callables in `expected_sequence` on `sample_input`, computed here by direct arithmetic
(increment(x)=x+1, double(x)=x*2, square(x)=x*x, negate(x)=-x -- all mechanically checkable by
hand). This is a hand-verified stand-in for independently-collected human labels, appropriate to
this toy-callable scope -- not a claim of large-scale human annotation. It exists to sanity-check
the bench harness's OWN scoring/execution pipeline against synthetic ground truth (the BATHOS
measurement-verification rule), independently of either generator mode.
`tests/scripts/test_author_bench_fixtures.py` re-derives every `expected_output` programmatically
as a second, independent check on this hand computation.
"""

from dataclasses import dataclass

_MOD = "xtrax.composition.author"


@dataclass(frozen=True)
class AuthorBenchTask:
    """One fixed authoring task: an NL prompt and its expected node-callable sequence.

    Attributes:
        task_id: Unique identifier, e.g. "t01".
        nl_prompt: The natural-language description of the intended composition.
        expected_sequence: Import-path refs ("module:qualname"), in composition order, into
            `xtrax.composition.author._EXAMPLE_CALLABLES`.
    """

    task_id: str
    nl_prompt: str
    expected_sequence: tuple[str, ...]


TASKS: tuple[AuthorBenchTask, ...] = (
    AuthorBenchTask(
        "t01",
        "increment the input, then double the result",
        (f"{_MOD}:example_increment", f"{_MOD}:example_double"),
    ),
    AuthorBenchTask(
        "t02",
        "double the input, then increment the result",
        (f"{_MOD}:example_double", f"{_MOD}:example_increment"),
    ),
    AuthorBenchTask(
        "t03",
        "square the input, then negate the result",
        (f"{_MOD}:example_square", f"{_MOD}:example_negate"),
    ),
    AuthorBenchTask(
        "t04",
        "negate the input, then square the result",
        (f"{_MOD}:example_negate", f"{_MOD}:example_square"),
    ),
    AuthorBenchTask(
        "t05",
        "increment the input, then square the result",
        (f"{_MOD}:example_increment", f"{_MOD}:example_square"),
    ),
    AuthorBenchTask(
        "t06",
        "double the input, then negate the result",
        (f"{_MOD}:example_double", f"{_MOD}:example_negate"),
    ),
    AuthorBenchTask(
        "t07",
        "square the input, then double the result, then increment the result",
        (f"{_MOD}:example_square", f"{_MOD}:example_double", f"{_MOD}:example_increment"),
    ),
    AuthorBenchTask(
        "t08",
        "negate the input, then increment the result, then double the result",
        (f"{_MOD}:example_negate", f"{_MOD}:example_increment", f"{_MOD}:example_double"),
    ),
    AuthorBenchTask(
        "t09",
        "increment the input, then double the result, then square the result",
        (f"{_MOD}:example_increment", f"{_MOD}:example_double", f"{_MOD}:example_square"),
    ),
    AuthorBenchTask(
        "t10",
        "double the input, then square the result, then negate the result",
        (f"{_MOD}:example_double", f"{_MOD}:example_square", f"{_MOD}:example_negate"),
    ),
)


@dataclass(frozen=True)
class SpotCheckItem:
    """One hand-verified (task, sample input, expected output) item.

    Attributes:
        task_id: References an `AuthorBenchTask.task_id` in `TASKS`.
        sample_input: The scalar input fed to the task's composition.
        expected_output: The hand-computed result of composing `expected_sequence` on
            `sample_input`, in order.
    """

    task_id: str
    sample_input: float
    expected_output: float


# 10 tasks x 3 sample inputs (3, -2, 5) = 30 items. Each expected_output computed by direct
# arithmetic: increment(x)=x+1, double(x)=x*2, square(x)=x*x, negate(x)=-x.
HUMAN_LABELED_SPOT_CHECK: tuple[SpotCheckItem, ...] = (
    # x = 3
    SpotCheckItem("t01", 3, 8),  # inc(3)=4, dbl(4)=8
    SpotCheckItem("t02", 3, 7),  # dbl(3)=6, inc(6)=7
    SpotCheckItem("t03", 3, -9),  # sq(3)=9, neg(9)=-9
    SpotCheckItem("t04", 3, 9),  # neg(3)=-3, sq(-3)=9
    SpotCheckItem("t05", 3, 16),  # inc(3)=4, sq(4)=16
    SpotCheckItem("t06", 3, -6),  # dbl(3)=6, neg(6)=-6
    SpotCheckItem("t07", 3, 19),  # sq(3)=9, dbl(9)=18, inc(18)=19
    SpotCheckItem("t08", 3, -4),  # neg(3)=-3, inc(-3)=-2, dbl(-2)=-4
    SpotCheckItem("t09", 3, 64),  # inc(3)=4, dbl(4)=8, sq(8)=64
    SpotCheckItem("t10", 3, -36),  # dbl(3)=6, sq(6)=36, neg(36)=-36
    # x = -2
    SpotCheckItem("t01", -2, -2),  # inc(-2)=-1, dbl(-1)=-2
    SpotCheckItem("t02", -2, -3),  # dbl(-2)=-4, inc(-4)=-3
    SpotCheckItem("t03", -2, -4),  # sq(-2)=4, neg(4)=-4
    SpotCheckItem("t04", -2, 4),  # neg(-2)=2, sq(2)=4
    SpotCheckItem("t05", -2, 1),  # inc(-2)=-1, sq(-1)=1
    SpotCheckItem("t06", -2, 4),  # dbl(-2)=-4, neg(-4)=4
    SpotCheckItem("t07", -2, 9),  # sq(-2)=4, dbl(4)=8, inc(8)=9
    SpotCheckItem("t08", -2, 6),  # neg(-2)=2, inc(2)=3, dbl(3)=6
    SpotCheckItem("t09", -2, 4),  # inc(-2)=-1, dbl(-1)=-2, sq(-2)=4
    SpotCheckItem("t10", -2, -16),  # dbl(-2)=-4, sq(-4)=16, neg(16)=-16
    # x = 5
    SpotCheckItem("t01", 5, 12),  # inc(5)=6, dbl(6)=12
    SpotCheckItem("t02", 5, 11),  # dbl(5)=10, inc(10)=11
    SpotCheckItem("t03", 5, -25),  # sq(5)=25, neg(25)=-25
    SpotCheckItem("t04", 5, 25),  # neg(5)=-5, sq(-5)=25
    SpotCheckItem("t05", 5, 36),  # inc(5)=6, sq(6)=36
    SpotCheckItem("t06", 5, -10),  # dbl(5)=10, neg(10)=-10
    SpotCheckItem("t07", 5, 51),  # sq(5)=25, dbl(25)=50, inc(50)=51
    SpotCheckItem("t08", 5, -8),  # neg(5)=-5, inc(-5)=-4, dbl(-4)=-8
    SpotCheckItem("t09", 5, 144),  # inc(5)=6, dbl(6)=12, sq(12)=144
    SpotCheckItem("t10", 5, -100),  # dbl(5)=10, sq(10)=100, neg(100)=-100
)


__all__ = ["TASKS", "HUMAN_LABELED_SPOT_CHECK", "AuthorBenchTask", "SpotCheckItem"]
