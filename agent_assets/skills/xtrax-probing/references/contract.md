# ProbeRecord Contract Reference

Owner modules: `src/xtrax/profiling/record.py` (schema + guards),
`src/xtrax/profiling/claims.py` (claim rules). When this document and the
code disagree, the code wins.

## Fields

Caller-supplied (the semantic content): `probe_id`, `stage`, `n_atoms`,
`platform` ("cpu" | "gpu"), `metrics`, `scopes`, `config`,
`attribution_method`.

Auto-captured at construction via default_factory (overridable explicitly --
e.g. synthetic test fixtures): `git_sha`, `timestamp`, `x64_enabled`,
`jax_version`, `jaxlib_version`, `xla_flags`, `device_kind`. A provenance
field a caller can forget is a provenance field that will be forgotten.

`metrics` holds floats only after construction: int and numeric-string inputs
are coerced; booleans, non-coercible values, NaN/inf raise
`ClaimValidityError` (bool rejection: JSON true/false would coerce to 1.0/0.0
and launder a flag into a citable metric).

## Construction Guards (`__post_init__`, all raise ClaimValidityError)

- `stage in {0,1,2,3}` (bool rejected -- True == 1).
- `stage >= 2` requires `platform == "gpu"` AND a `device_kind`: Stage-2+
  records are GPU-measured by definition.
- `n_atoms > 0` (bool rejected).
- `scopes` values: None OR an `(exclusive_seconds, n_occurrences)` pair with
  finite non-negative numeric seconds and integer occurrences >= 1. This
  closes the from_json hole where JSON's bare NaN/Infinity tokens smuggled
  corrupt scope rows into claim-gated reports.
- Attribution coverage: every MEASURED (non-None) scope label must appear in
  `attribution_method`; no attribution may name a label scopes lacks.
  Attributing a null label is allowed and meaningful ("known named_scope that
  never fired"). `scopes=None` tolerates any attribution map (inert).
- `attribution_method` values restricted to "named_scope" | "op_name".

## Write / Read

`write()` is atomic (tmp + os.replace): a crash mid-write cannot leave a
truncated record. `from_json()` fails closed in a deliberate order: unknown
fields -> contract-version major mismatch -> hard-missing default_factory
fields -> malformed values. Each diagnostic names the actual problem.

## Claim Classes and Floors

| class | single-record floor (`permitted_claims`) | set rules (`assert_claim_supported`) |
|---|---|---|
| STRUCTURAL | always | -- |
| DISPATCH_COUNT | stage >= 1 | stage >= 1 re-enforced on the assert path |
| TERM_RANKING | never granted alone | all sources stage >= 2 (GPU); >=2 attributed scopes carrying `total_step_seconds`; verifiable shas; 5-field unanimity |
| END_TO_END | never granted alone | target_n_atoms > 0 (or allow_no_target=True); target/min(n_atoms) <= SCALE_EXTRAPOLATION_LIMIT; same sha/unanimity rules |

The stage floors are enforced TWICE by design: `permitted_claims` for display
and filtering, and again inside `assert_claim_supported` -- the authoritative
path must not depend on callers having consulted the filter first.

Unanimity fields: x64_enabled, xla_flags, device_kind, platform, git_sha.
Measured XLA_FLAGS differences alone produced a 1170x throughput change on
prolix's Blackwell nodes -- this guard exists because of that.

Verifiable git_sha = non-empty, not "unknown", not "-dirty", not
"-unverified". `_capture_git_sha` honors XTRAX_GIT_SHA, then repo-root
`.git_sha`, then git rev-parse (suffixing -dirty/-unverified on any failure;
an empty `.git_sha` file reads as "unknown").

`select_sources` (metric-keyed, then stage-maximal) raises rather than
returning empty, distinguishing missing-metric from missing-scope-attribution
causes so the fix message is actionable.

## Contract Versioning

`CONTRACT_VERSION` bumps follow the dataclass bump rule: adding/removing a
field with a default_factory = MAJOR (old records unreadable); anything else
= MINOR/PATCH. The prolix->xtrax port changed no field set and kept "3.0".

`from_json` rejects records whose major version differs from the running
contract BEFORE inspecting fields, so version skew is never misdiagnosed as
corruption.

## Port Fidelity

The package was upstreamed from prolix `scripts/profiling`
(wt-20260807-132628). An AST-level fidelity audit (scope doc, 2026-08-25)
confirmed zero undocumented semantic drift: trace.py/__init__.py identical
after normalization; all divergent defs map to recorded decisions (env-var
rename, parents[3] layout, slots dataclass, future-annotations removal,
metrics annotation widening, D6 discovery root, datetime UTC idiom).
