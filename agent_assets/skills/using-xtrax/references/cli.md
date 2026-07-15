> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# CLI Layer (E2/E3) — Tyro-delegated verbs: plan/explain/export/run/resume/sweep + graph-validate/graph-plan/graph-author

> **Availability**: six verbs shipped in the 0.3.0 release (E2: `plan`/`explain`/`export`; E3: `run`/`resume`/`sweep`). Three more — `graph-validate`/`graph-plan`/`graph-author` (T1-10/T1-11/T1-12) — are **unreleased, main-only** (no CHANGELOG entry yet, not in the 0.4.0a5 wheel), same convention as this doc's other main-only flags (e.g. the `io_callback` shim, `axis_boundaries_by_name`). Verify against `REGISTRY` directly (`src/xtrax/cli/registry.py`) before relying on a verb count — this doc is a map, not the territory.

#### Verb Registry (Tyro-delegated)

All CLI verbs are registered in `REGISTRY` — a single dict mapping verb name → `(ArgsClass, run_fn)`:

```python
from xtrax.cli.registry import REGISTRY  # verify: src/xtrax/cli/registry.py:41-51

# REGISTRY keys (E2/E3 — 0.3.0 release):
#   "plan"    → (PlanArgs, run_plan)       — infer_bundle + BatchPlanner, print summary
#   "explain" → (ExplainArgs, run_explain) — infer_bundle + plan + explain_plan + emit
#   "export"  → (ExportArgs, run_export)   — export plan artifacts
#   "run"     → (RunArgs, run_run)         — load_config → run_from_config → Engine.fit_sync
#   "resume"  → (ResumeArgs, run_resume)   — read manifest → reconstruct state from latest ckpt → train N more epochs
#   "sweep"   → (SweepArgs, run_sweep)     — sequential in-process grid search over a sweep TOML

# REGISTRY keys (T1-10/11/12 — unreleased, main-only):
#   "graph-validate" → (GraphValidateArgs, run_graph_validate) — load <ir.json>, run validate_graph, write audit_verdict back
#   "graph-plan"     → (GraphPlanArgs, run_graph_plan)         — load <ir.json>, resolve a node's callable_ref, plan it via plan_from_fn
#   "graph-author"   → (GraphAuthorArgs, run_graph_author)     — free-generate a candidate IR via TemplateGenerator, validate in-process, write it
```

`entrypoint.main()` builds a tyro subcommand dict from `REGISTRY` and dispatches the parsed `ArgsClass` instance to its `run_fn`. Verify: `src/xtrax/cli/entrypoint.py:19-48`

#### `xtrax run config.toml` Flow

End-to-end training from a TOML file:

```
config.toml
  → load_config(path)          # tomllib parse + validation  — verify: src/xtrax/cli/config.py:38-56
  → TrainConfig                # cli-private dataclass       — verify: src/xtrax/cli/config.py:15-29
  → run_from_config(cfg)       # cli-private glue            — verify: src/xtrax/cli/run.py:38-80
      → resolve model/optimizer/loss/data via load_fn (import-path strings)
      → init_state(model, optimizer, seed)   # public API    — verify: src/xtrax/training/state.py:9-20
      → config_hash(cfg_dict)  # run_id derivation           — verify: src/xtrax/cli/hash.py:7-20
      → write_manifest(...)    # always before fit_sync      — verify: src/xtrax/cli/manifest.py:56-77
      → Engine(Trainer(...)).fit_sync(state, data, ...)
```

CLI entry: `run_run(RunArgs(config="config.toml"))` catches `ConfigError` and exits with a clean message. Verify: `src/xtrax/cli/run_verb.py:14-20`

#### `xtrax resume <run-id> --epochs N` Flow

Resume a prior run from its latest orbax checkpoint, training N **additional** epochs into a new sibling run dir:

```
xtrax resume <run-id> --epochs N [--manifest-path PATH]
  → read_manifest(run_id)                 # locate the run's manifest.json
  → resolve_components(...)               # re-resolve model/optimizer/loss/data from import paths
  → load_checkpoint(...)                  # reconstruct ResumableState from latest orbax ckpt
  → write_manifest_dict(...)              # new sibling run dir under .xtrax/runs/
  → Engine(Trainer(...)).fit_sync(...)
```

`ResumeArgs`: positional `run_id`, required `epochs: int`, optional `manifest_path` (if the run dir was moved). Raises `ResumeError` (subclasses `CLIError`) on a missing/invalid manifest or checkpoint. Verify: `src/xtrax/cli/resume_verb.py:18-30`

#### `xtrax sweep sweep_config.toml` Flow

Sequential in-process grid search. The sweep config is a normal training TOML plus a `[sweep.axes]` section whose leaves are **lists** — the grid is the cartesian product, and each combination overrides the base config for one run:

```toml
# ... normal [model]/[optimizer]/[loss]/[data] sections ...

[sweep.axes]
seed = [42, 43]
optimizer.kwargs.peak_lr = [1e-3, 3e-4]   # nested keys via dotted tables or nesting
```

Properties (verify: `src/xtrax/cli/sweep_verb.py`):
- Sweep manifest written incrementally and atomically per combination (`tempfile.mkstemp` + `os.replace` before each run executes); per-run fault tolerance (one failed combination doesn't kill the sweep).
- JAX compilation cache reused across combinations (single process, sequential — isolates compilation and execution memory via `gc` between runs).
- 🚫 HALTS: `ConfigError` if `[sweep]` is not a table or any `sweep.axes` leaf is not a list.

`SweepArgs`: positional `config_path` only. Verify: `src/xtrax/cli/sweep_verb.py:34-37`

#### `xtrax graph-validate <ir.json>` Flow (unreleased, main-only, T1-10)

Validates a D4 IR document in place and writes the audit verdict back into it:

```
xtrax graph-validate <ir.json>
  → load_graph(path)               # parse D4 IR document          — verify: src/xtrax/composition/serialize.py
  → validate_graph(graph, root=cwd)  # deterministic validation gate — verify: src/xtrax/composition/validate.py
  → dump_graph(result.graph, path)   # write audit_verdict back, same path
  → print JSON envelope {schema_version, failure_count, failures[]}
```

`GraphValidateArgs`: positional `ir_path` only. Malformed input (missing/unknown `schema_version`, unresolvable `callable_ref`) raises `SystemExit` with a clean message. Any node not `verdict=PASS` exits 1 after printing the envelope. Registered as the flat verb `graph-validate` — `entrypoint.py`'s tyro dispatch is a flat `dict[str, (ArgsClass, run_fn)]` with no nested-subcommand support, so this is not the DAG doc's informal two-word `graph validate`. Verify: `src/xtrax/cli/graph_verb.py`

#### `xtrax graph-plan <ir.json> <node-id> [--shapes ...]` Flow (unreleased, main-only, T1-11)

Resolves a named node's `callable_ref` from a D4 IR document and plans it — the CLI-consumed half of AC1's graph→plan parity proof:

```
xtrax graph-plan <ir.json> <node_id> [--shapes "x=(4,)f32"]
  → load_graph(path)                       # same loader graph-validate uses
  → nodes_by_id[node_id].callable_ref       # CLIError if node_id not found (lists available ids)
  → plan_from_fn(callable_ref, shapes)      # same plan_from_fn helper run_plan's bare --fn/--shapes path uses
  → print_plan_summary(plan)
```

`GraphPlanArgs`: positional `ir_path`, required `node_id`, optional `shapes` (default `""`; see `xtrax.cli.shapes.parse_shapes` grammar). A node's `callable_ref` post-`load_graph` resolves to the identical live function object `load_fn` would resolve from a bare `module.path:symbol` string — both use the same convention, so this path is provably convergent with `run`'s `--fn` resolution, not just coincidentally similar. Verify: `src/xtrax/cli/graph_plan_verb.py`

#### `xtrax graph-author <out.json> [--seed N] [--num-nodes N]` Flow (unreleased, main-only, T1-12)

The default generate-then-validate authoring front-end — free-generates a candidate graph and validates it in-process before writing:

```
xtrax graph-author <out_path> [--seed 0] [--num-nodes 3]
  → TemplateGenerator().generate(seed, num_nodes)   # deterministic free-generation
  → validate_graph(graph, root=cwd)                 # same gate graph-validate uses, run in-process
  → dump_graph(result.graph, out_path)
  → print JSON envelope {schema_version, failure_count, failures[]}
```

`GraphAuthorArgs`: positional `out_path`, `seed: int = 0`, `num_nodes: int = 3`. "Authors ≥1 graph passing graph-validate" is enforced directly here as an in-process assertion (`SystemExit(1)` on any non-`PASS` verdict), not left for a caller to separately verify by running `graph-validate` afterward. An invalid `num_nodes` or unwritable `out_path` raises `SystemExit` with a clean message. Verify: `src/xtrax/cli/graph_author_verb.py`

#### Key Types

| Symbol | Module | Role |
|--------|--------|------|
| `TrainConfig` | `xtrax.cli.config` | Parsed training config (`schema_version`, `model`, `optimizer`, `loss`, `data`, `seed`, `num_epochs`) |
| `ConfigError` | `xtrax.cli.config` | Invalid/incomplete TOML; subclasses `CLIError` |
| `load_config` | `xtrax.cli.config` | Parse + validate TOML path → `TrainConfig`; composed from `xtrax.config`'s primitives (below) |
| `init_state` | `xtrax.training` | **Public API** — build `ResumableState` from model + optimizer + seed |
| `config_hash` | `xtrax.cli.hash` | cli-private — stable 12-char hex hash for run-id derivation |
| `write_manifest` | `xtrax.cli.manifest` | cli-private — always-write `manifest.json` under `.xtrax/runs/<run_id>/` |
| `load_fn` | `xtrax.cli` (also `xtrax.cli.loader`) | **Stable public API** — domain-agnostic `module.path:symbol` → callable resolver; safe for downstream packages to import directly |
| `CLIError` / `CLIImportError` | `xtrax.cli` (also `xtrax.cli.errors`) | **Stable public API** — downstream CLIs may subclass `CLIError` for their own fail-loud error types (mirrors `ConfigError`/`ResumeError` in-repo) |
| `REGISTRY` | `xtrax.cli` (also `xtrax.cli.registry`) | **Stable public API** (keys/shape only) — `verb_name -> (ArgsClass, run_fn)`; see the REGISTRY-composition pattern below |
| `load_toml_document` | `xtrax.config` | **Stable public API** — domain-agnostic TOML parse, wraps IO/decode errors into a caller-supplied `error_cls` |
| `require_sections` | `xtrax.config` | **Stable public API** — presence check naming *every* missing section, not just the first |
| `require_field` | `xtrax.config` | **Stable public API** — extract + validate a field against an arbitrary predicate |
| `check_schema_version` / `classify_schema_version` | `xtrax.config` | **Stable public API** — schema-version validation; `classify_schema_version` is the public extension seam for future status kinds |

`init_state` is re-exported from `xtrax.training` (`__all__` at `src/xtrax/training/__init__.py:14`). `TrainConfig`/`load_config`/`ConfigError` stay in `xtrax.cli.config` — cli-private, not top-level `xtrax` exports, and training-shaped (not suitable for a non-training consumer to import directly). By contrast, `load_fn`/`CLIError`/`CLIImportError`/`REGISTRY` ARE declared public at `xtrax.cli.__all__`, and `xtrax.config`'s four primitives are a fully domain-agnostic top-level module — a downstream consumer that needs training-config-shaped TOML loading should compose `xtrax.config`'s primitives directly for its own dataclass shape (see the Minimal `xtrax.config` Usage example below), not mirror `TrainConfig`'s pattern by hand and not import `TrainConfig` itself.

#### Minimal `xtrax.config` Usage (domain-agnostic, not training-shaped)

```python
from dataclasses import dataclass
from xtrax.config import load_toml_document, check_schema_version, require_sections, require_field

class InferConfigError(Exception):
    """A downstream package's own error type -- xtrax.config never hardcodes one."""

@dataclass
class InferConfig:
    schema_version: int
    model: dict
    checkpoint: dict

def load_infer_config(path: str) -> InferConfig:
    raw = load_toml_document(path, InferConfigError)
    check_schema_version(raw, current=1, error_cls=InferConfigError)
    require_sections(raw, ("model", "checkpoint"), InferConfigError)
    return InferConfig(schema_version=raw["schema_version"], model=raw["model"], checkpoint=raw["checkpoint"])
```

Verify: `src/xtrax/config.py`; `xtrax.cli.config.load_config` is the dog-fooded reference usage (`src/xtrax/cli/config.py`). Spec: `.praxia/docs/specs/260715_generic-fail-loud-toml-to-dataclass-conf.md`.

#### Minimal `config.toml` Skeleton

Each section uses import-path `path`/`factory` keys plus optional `kwargs`. Verify against `tests/cli/test_config.py:16-37`:

```toml
schema_version = 1
seed = 42
num_epochs = 3

[model]
path = "mylib.models:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = { learning_rate = 1e-3, total_steps = 300 }

[loss]
path = "mylib.losses:mse_loss"
kwargs = {}

[data]
factory = "mylib.data:make_dataset"
kwargs = {}
batch_size = 4
```

🚫 HALTS: Missing `schema_version` or any of `[model]`, `[optimizer]`, `[loss]`, `[data]` raises `ConfigError`, naming **every** missing section (not just the first).  
🚫 HALTS: `num_epochs` must be a positive int; `seed` must be an int.  
Enforcement: `src/xtrax/cli/config.py` (composed from `xtrax.config`'s `check_schema_version`/`require_sections`/`require_field`).

#### Tyro-Free Import Rule

`import xtrax.cli` must **not** pull `tyro` at module level (AC2 import isolation):

- `xtrax.cli.__init__` exports `CLIError`, `CLIImportError`, `ShapeParseError`, `load_fn`, `REGISTRY`, and a lazy `main()` that imports `entrypoint` on demand. `REGISTRY` is exposed via a PEP 562 module-level `__getattr__` — accessing it lazily imports `xtrax.cli.registry` (and thus all 9 built-in verb modules) on demand, so a bare `import xtrax.cli` stays as lightweight as before this export was added. Verify: `src/xtrax/cli/__init__.py`
- `entrypoint.main()` imports `tyro` **inside** the function body. Verify: `src/xtrax/cli/entrypoint.py:30-31`

Test pattern (mirrors E2 isolation tests): `assert "tyro" not in sys.modules` immediately after `import xtrax.cli`.

#### REGISTRY Composition: Reusing xtrax's Verbs in Your Own CLI

A downstream package building its own tyro-dispatched CLI (rather than getting a verb hosted through `xtrax`'s own binary — that entry-points-plugin approach was considered and explicitly **deferred**, see `.praxia/docs/specs/260715_entry-points-based-xtrax-cli-verb-regist.md`) can reuse xtrax's own verbs today, with zero xtrax code changes:

```python
import tyro
from xtrax.cli import REGISTRY

my_verbs = {"infer": (InferArgs, run_infer)}  # your own package's verbs
merged = {**REGISTRY, **my_verbs}

subcommands = {name: args_cls for name, (args_cls, _fn) in merged.items()}
selected = tyro.extras.subcommand_cli_from_dict(subcommands)
for name, (args_cls, run_fn) in merged.items():
    if args_cls is type(selected):
        run_fn(selected)
```

**Stability boundary:** `REGISTRY`'s **keys and dict shape** (`verb_name -> (ArgsClass, run_fn)`) are a stable, documented contract. The `ArgsClass`/`run_fn` internal typing is **provisional**, not independently versioned — don't rely on a specific verb's `ArgsClass` field set staying fixed across xtrax releases beyond what its own docs promise.

🚫 HALTS: a verb-name or `ArgsClass` field-name collision between `REGISTRY` and your own verbs is your responsibility to avoid — `tyro.extras.subcommand_cli_from_dict` does not detect or warn on one. Verify: `tests/cli/test_registry_composition.py`.
