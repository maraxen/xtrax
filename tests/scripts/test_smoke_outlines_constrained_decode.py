"""Tests for T1-13's `prompt` override on T1-12's smoke script (#3068).

Fully mocked via `sys.modules` injection rather than `pytest.importorskip("outlines")` -- this
lets the test verify the prompt-plumbing logic in normal CI (no `authoring-smoke` extra
installed) instead of being silently skipped whenever the heavy deps aren't present.
"""

import sys
import types
from typing import Any

import pytest

from scripts.smoke_outlines_constrained_decode import DEFAULT_PROMPT, run_smoke


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, prompt: str, schema: Any, max_new_tokens: int = 900) -> str:
        self.calls.append(prompt)
        return '{"schema_version": 1, "nodes": [], "edges": []}'


@pytest.fixture
def fake_outlines_stack(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    model = _FakeModel()

    fake_outlines = types.ModuleType("outlines")
    fake_outlines.from_transformers = lambda *_args, **_kwargs: model  # type: ignore[attr-defined]

    fake_outlines_types = types.ModuleType("outlines.types")
    fake_outlines_types.JsonSchema = lambda schema_str: schema_str  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name: object()
    )
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name: object()
    )

    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.types", fake_outlines_types)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return model


class TestRunSmokePromptOverride:
    def test_default_prompt_used_when_none_given(self, fake_outlines_stack: _FakeModel) -> None:
        run_smoke()
        assert fake_outlines_stack.calls == [DEFAULT_PROMPT]

    def test_custom_prompt_overrides_default(self, fake_outlines_stack: _FakeModel) -> None:
        custom = "Author a graph that increments then doubles the input."
        run_smoke(prompt=custom)
        assert fake_outlines_stack.calls == [custom]
        assert custom != DEFAULT_PROMPT
