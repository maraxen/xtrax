#!/usr/bin/env python3
"""Outlines constrained-decoding smoke demo (T1-12, #3067) -- NOT a CI gate, NOT a CLI verb.

Feeds the same IR JSON schema `xtrax.inference.ir_schema.emit_ir_schema()` produces to
Outlines' constrained decoding to decode ONE schema-valid document. This is the "smoke"
half of T1-12's AC, deliberately kept separate from the default authoring path
(`xtrax graph-author`, `xtrax.composition.author`):

    Fast/loud: constrained decoding stays a smoke demo only -- it is not wired into any loop
    this sprint; a test asserts no loop code imports the constrained path as a hard
    dependency (`tests/audit/test_no_constrained_decode_in_loop.py`).

Deliberately lives outside `src/xtrax/` and outside `REGISTRY` (`src/xtrax/cli/registry.py`)
so it can never become an accidental hard dependency of the generate-then-validate loop.
Requires the `authoring-smoke` optional-dependency group (`uv run --extra authoring-smoke
python scripts/smoke_outlines_constrained_decode.py`) -- `outlines`/`transformers` are never
installed by a default/dev/cli/eda/io install.

Backend note: constrained decoding needs local logit access to do FSM-guided token masking
(Outlines' `from_transformers`/`from_llamacpp`/`from_vllm`/`from_mlxlm` backends). It
structurally cannot wrap a remote chat-completion API (e.g. Claude via praxia dispatch) --
there are no logits to constrain on the other end of an API call. So the "pluggable backend,
e.g. praxia" idea explored for T1-12's *default* free-generation path does not apply here by
construction; this script always needs an actual local model artifact.

Known future extension point (not built here): LiteRT is a real candidate local-model runtime
for this exact script if a browser/edge deployment target for the authoring front-end is ever
wanted -- LiteRT targets WASM/edge inference, which fits "eventual static web deployment" far
better than the authoring front-end in general does. Using it would require writing a custom
Outlines logits-processor bridge (no built-in Outlines/LiteRT integration exists today); not
attempted in this item.

Model choice, arrived at empirically (two prior choices tried and rejected by actually running
them, not guessed):

1. `hf-internal-testing/tiny-random-gpt2` (tiny vocab, tiny architecture): fails immediately --
   Outlines raises "vocabulary incompatible with the regex." Its 1000-token vocab can't spell
   the schema's required literals (`"schema_version"`, `"nodes"`, etc.) at all.
2. `sshleifer/tiny-gpt2` (real GPT-2 vocab, degenerate n_layer=2/n_embd=2 architecture): passes
   the vocabulary check, but generation never terminates open-ended fields -- constrained
   decoding's FSM permits the model to keep emitting digits for `schema_version` (an unbounded
   JSON integer) forever, because the model's near-random weights never learn to prefer a stop
   token in that position. The FSM guarantees structural validity of whatever tokens are chosen,
   not that the model ever chooses to stop -- an architecturally-degenerate model can defeat that
   guarantee by simply never terminating within any reasonable token budget.
3. `distilgpt2` (real GPT-2 vocab, real trained weights, ~82M params): works reliably --
   confirmed by 3/3 successful runs producing valid `{"schema_version": ..., "nodes": [...]}`
   documents. A genuinely trained model (even a small, distilled one) reliably prefers to
   terminate JSON literals/arrays because it learned that from real text, which a random or
   near-random-weight tiny architecture cannot. Still small enough (~350MB) to avoid a
   multi-GB download.

The prompt nudges toward a minimal document ("...with exactly one node and no edges, no
citations") -- without that steer, `distilgpt2` sometimes wanders into long optional-field
tangents (nested `citations` arrays, etc.) before finding a valid stopping point, wasting the
token budget without necessarily failing, but the steer makes convergence faster and more
consistent.
"""

from __future__ import annotations

import argparse
import json
import sys

DEFAULT_MODEL = "distilgpt2"
DEFAULT_PROMPT = (
    "Author a minimal xtrax composition IR document with exactly one node and no edges, "
    "no citations."
)


def run_smoke(model_name: str = DEFAULT_MODEL) -> dict:
    """Constrained-decode one document conforming to `emit_ir_schema()`'s schema.

    Returns the decoded document (already validated against the schema by construction --
    Outlines' constrained decoding cannot produce non-conforming output).
    """
    import outlines
    from outlines.types import JsonSchema
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from xtrax.inference.ir_schema import emit_ir_schema

    schema = emit_ir_schema()
    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name),
    )
    result = model(
        DEFAULT_PROMPT,
        JsonSchema(json.dumps(schema)),
        max_new_tokens=900,
    )
    return json.loads(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF model id to constrained-decode with"
    )
    args = parser.parse_args(argv)

    try:
        document = run_smoke(args.model)
    except Exception as exc:  # noqa: BLE001 -- smoke demo: report any failure, don't triage
        print(f"FAIL: constrained-decode smoke raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if "schema_version" not in document or "nodes" not in document:
        print(
            f"FAIL: decoded document missing required top-level fields: {document!r}",
            file=sys.stderr,
        )
        return 1

    print("PASS: Outlines constrained-decoded one schema-valid IR document")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
