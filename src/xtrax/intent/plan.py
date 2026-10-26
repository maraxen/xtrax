"""Recorded site decisions for one intent-resolution context.

A :class:`Plan` is a frozen, hashable tuple of :class:`SiteDecision` values.
Strategies serialise as ``{"type": <ClassName>, ...fields}``. ``Context`` is
written only as ``repr``; :meth:`Plan.from_json` takes the live context back.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from xtrax.intent.context import Context
from xtrax.tiling.strategy import AxisStrategy, ChunkedMap, Scan, StaticWhile, Vmap, WhileCarry

PLAN_SCHEMA_VERSION = 1

_STRATEGY_FIELDS: dict[type[Any], tuple[str, ...]] = {
    Vmap: (),
    ChunkedMap: ("batch_size",),
    Scan: ("ordered_sinks",),
    StaticWhile: ("length",),
    WhileCarry: (),
}


@dataclass(frozen=True)
class SiteKey:
    """Identity of one intent call on one resolution path.

    ``label`` is the user ``name`` or the caller's ``"module:line"``.
    ``ordinal`` separates repeated calls that share a parent, path, and label.

    Attributes
    ----------
    parent : SiteKey or None
        Enclosing site. ``None`` at the root.
    path : {"primal", "diff"}
        Resolution path this call was entered on.
    label : str
        User name or source location of the intent call.
    ordinal : int
        Index among calls with the same parent, path, and label.
    """

    parent: "SiteKey | None"
    path: Literal["primal", "diff"]
    label: str
    ordinal: int

    def __str__(self) -> str:
        """Render ``path/label#ordinal``, prefixed by the parent when nested.

        Parameters
        ----------
        self : SiteKey
            Key being rendered. A nested key prefixes ``str(parent)``.

        Returns
        -------
        str
            ``path/label#ordinal`` when ``parent`` is ``None``. Otherwise
            ``str(parent)/path/label#ordinal``.
        """
        own = f"{self.path}/{self.label}#{self.ordinal}"
        if self.parent is None:
            return own
        return f"{self.parent}/{own}"


def canonical(key: SiteKey) -> SiteKey:
    """Return ``key`` with ``path="primal"`` at every level of the parent chain.

    Parameters
    ----------
    key : SiteKey
        Key to rewrite. Parents are rewritten first.

    Returns
    -------
    SiteKey
        The primal-chain key for this visit.
    """
    parent = None if key.parent is None else canonical(key.parent)
    return SiteKey(parent=parent, path="primal", label=key.label, ordinal=key.ordinal)


@dataclass(frozen=True)
class SiteDecision:
    """Strategy chosen for one site, on the primal and differentiated paths.

    ``diff is None`` means there is no differentiated decision. ``diff_reason``
    still says why.

    Attributes
    ----------
    key : SiteKey
        Site this decision belongs to.
    kind : str
        Site kind, such as ``"loop"`` or ``"map"``.
    primal : AxisStrategy or StaticWhile
        Strategy for the primal path.
    diff : AxisStrategy or None
        Strategy for the differentiated path, or ``None`` when that path
        has no decision.
    primal_reason : str
        Why ``primal`` was chosen.
    diff_reason : str
        Why ``diff`` was chosen, or why there is no differentiated decision.
    warnings : tuple of str
        Non-fatal notes attached to the decision.
    """

    key: SiteKey
    kind: str
    primal: AxisStrategy | StaticWhile
    diff: AxisStrategy | None
    primal_reason: str
    diff_reason: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """Versioned, hashable record of every site decision for one context.

    Attributes
    ----------
    schema_version : int
        Schema of the JSON form. Writers use :data:`PLAN_SCHEMA_VERSION`.
    context : Context
        Constraints these decisions were resolved under. Not reconstructed
        from JSON; passed back into :meth:`from_json`.
    decisions : tuple of SiteDecision
        One decision per visited site.
    warnings : tuple of str
        Notes about the plan as a whole, such as the arguments a partial
        differentiation could not linearize. Empty when there are none. A
        plan written before this field existed loads with ``()``.
    """

    schema_version: int
    context: Context
    decisions: tuple[SiteDecision, ...]
    warnings: tuple[str, ...] = ()

    def by_key(self) -> Mapping[SiteKey, SiteDecision]:
        """Index decisions by site key.

        Parameters
        ----------
        self : Plan
            Plan whose decisions are indexed.

        Returns
        -------
        Mapping
            Each :class:`SiteKey` mapped to its :class:`SiteDecision`.

        Raises
        ------
        ValueError
            Two decisions share a :class:`SiteKey`. The message names that key.
        """
        indexed: dict[SiteKey, SiteDecision] = {}
        for decision in self.decisions:
            key = decision.key
            if key in indexed:
                raise ValueError(f"duplicate SiteKey {key}")
            indexed[key] = decision
        return indexed

    def table(self) -> str:
        """Render one row per site.

        Parameters
        ----------
        self : Plan
            Plan being printed.

        Returns
        -------
        str
            Header plus one tab-separated row per decision: key, kind,
            primal, diff, primal reason, diff reason, warnings. Each plan-level
            warning follows as a ``warning: <text>`` line.
        """
        rows = ["key\tkind\tprimal\tdiff\tprimal_reason\tdiff_reason\twarnings"]
        for decision in self.decisions:
            warnings = ", ".join(decision.warnings)
            rows.append(
                "\t".join(
                    (
                        str(decision.key),
                        decision.kind,
                        _strategy_cell(decision.primal),
                        _strategy_cell(decision.diff),
                        decision.primal_reason,
                        decision.diff_reason,
                        warnings,
                    )
                )
            )
        rows.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(rows)

    def to_json(self) -> str:
        """Serialise the plan.

        ``context`` is ``repr(context)`` only. Strategies are
        ``{"type": <ClassName>, ...fields}`` for ``Vmap``, ``ChunkedMap``,
        ``Scan`` (``ordered_sinks`` only), ``StaticWhile``, and ``WhileCarry``
        (no fields; ``body``, ``cond``, and ``init`` must be unset).

        Parameters
        ----------
        self : Plan
            Plan being serialised.

        Returns
        -------
        str
            JSON text with default :func:`json.dumps` separators.

        Raises
        ------
        ValueError
            A ``Scan`` has a non-``None`` ``transition`` or ``init``, a
            ``WhileCarry`` has a non-``None`` ``body``, ``cond``, or ``init``,
            or a strategy type has no JSON form. The message names the type.
        """
        payload = {
            "schema_version": self.schema_version,
            "context": repr(self.context),
            "decisions": [_decision_to_json(decision) for decision in self.decisions],
            "warnings": list(self.warnings),
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, s: str, *, context: Context) -> "Plan":
        """Rebuild a plan from :meth:`to_json`, using the supplied context.

        Parameters
        ----------
        cls : type[Plan]
            Plan class.
        s : str
            JSON text from :meth:`to_json`.
        context : Context
            Live context. The ``repr`` stored in ``s`` is not read back.

        Returns
        -------
        Plan
            Decisions from ``s`` paired with ``context``.

        Raises
        ------
        ValueError
            ``schema_version`` is not :data:`PLAN_SCHEMA_VERSION`, or the
            payload is not an object with ``decisions``, keys, and strategies,
            or ``warnings`` is present and not a list.
            The message names the field. An unknown strategy type is named too.
        json.JSONDecodeError
            ``s`` is not JSON text.
        """
        payload = json.loads(s)
        if not isinstance(payload, dict):
            raise ValueError("plan JSON is not an object")
        version = payload.get("schema_version")
        if version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {version!r} != supported schema_version {PLAN_SCHEMA_VERSION}"
            )
        if "decisions" not in payload:
            raise ValueError("missing 'decisions'")
        raw_decisions = payload["decisions"]
        if not isinstance(raw_decisions, list):
            raise ValueError("'decisions' is not a list")
        decisions = tuple(_decision_from_json(item) for item in raw_decisions)
        raw_warnings = payload.get("warnings", [])
        if not isinstance(raw_warnings, list):
            raise ValueError("'warnings' is not a list")
        return cls(version, context, decisions, tuple(raw_warnings))


def _strategy_cell(strategy: AxisStrategy | StaticWhile | None) -> str:
    """Short table cell for a strategy, or ``"None"``."""
    if strategy is None:
        return "None"
    name = type(strategy).__name__
    if isinstance(strategy, ChunkedMap):
        return f"{name}(batch_size={strategy.batch_size})"
    if isinstance(strategy, StaticWhile):
        return f"{name}(length={strategy.length})"
    if isinstance(strategy, Scan):
        return f"{name}(ordered_sinks={strategy.ordered_sinks})"
    return name


def _strategy_to_json(strategy: AxisStrategy | StaticWhile) -> dict[str, Any]:
    """Serialise one supported strategy.

    ``Scan`` keeps ``ordered_sinks`` only. ``WhileCarry`` keeps no fields.

    Raises
    ------
    ValueError
        ``Scan.transition`` or ``Scan.init`` is set, ``WhileCarry.body``,
        ``WhileCarry.cond``, or ``WhileCarry.init`` is set, or ``strategy``
        is not one of ``Vmap``, ``ChunkedMap``, ``Scan``, ``StaticWhile``,
        ``WhileCarry``.
    """
    if isinstance(strategy, Scan):
        callable_set = strategy.transition is not None or strategy.init is not None
        if callable_set:
            raise ValueError(
                "Scan cannot be serialised when transition or init is set "
                "(callables cannot be serialised)"
            )
    if isinstance(strategy, WhileCarry):
        callable_set = (
            strategy.body is not None or strategy.cond is not None or strategy.init is not None
        )
        if callable_set:
            raise ValueError(
                "WhileCarry cannot be serialised when body, cond, or init is set "
                "(callables cannot be serialised)"
            )
    fields = _STRATEGY_FIELDS.get(type(strategy))
    if fields is None:
        raise ValueError(f"cannot serialise strategy type {type(strategy).__name__}")
    payload: dict[str, Any] = {"type": type(strategy).__name__}
    for name in fields:
        payload[name] = getattr(strategy, name)
    return payload


_STRATEGY_TYPES: dict[str, type[Any]] = {
    "Vmap": Vmap,
    "ChunkedMap": ChunkedMap,
    "Scan": Scan,
    "StaticWhile": StaticWhile,
    "WhileCarry": WhileCarry,
}


def _object(payload: Any, what: str) -> dict[str, Any]:
    """Return ``payload`` when it is a JSON object."""
    if not isinstance(payload, dict):
        raise ValueError(f"{what} is not an object")
    return payload


def _field(payload: Mapping[str, Any], name: str) -> Any:
    """Return ``payload[name]`` or raise ``ValueError`` naming ``name``."""
    if name not in payload:
        raise ValueError(f"missing {name!r}")
    return payload[name]


def _strategy_from_json(payload: Any) -> AxisStrategy | StaticWhile:
    """Inverse of :func:`_strategy_to_json`.

    Raises
    ------
    ValueError
        ``type`` is missing or not a supported strategy, or a required
        strategy field is missing. The message names the field or the type.
    """
    body = _object(payload, "strategy")
    type_name = _field(body, "type")
    strategy_type = _STRATEGY_TYPES.get(type_name) if isinstance(type_name, str) else None
    if strategy_type is None:
        raise ValueError(f"unknown strategy type {type_name!r}")
    kwargs = {name: _field(body, name) for name in _STRATEGY_FIELDS[strategy_type]}
    return strategy_type(**kwargs)


def _key_to_json(key: SiteKey) -> dict[str, Any]:
    """Serialise a site key, including its parent chain."""
    parent = None if key.parent is None else _key_to_json(key.parent)
    return {"parent": parent, "path": key.path, "label": key.label, "ordinal": key.ordinal}


def _key_from_json(payload: Any) -> SiteKey:
    """Inverse of :func:`_key_to_json`.

    Raises
    ------
    ValueError
        ``payload`` is not an object, a key field is missing, or ``path``
        is not ``"primal"`` or ``"diff"``.
    """
    body = _object(payload, "key")
    parent_payload = _field(body, "parent")
    parent = None if parent_payload is None else _key_from_json(parent_payload)
    path = _field(body, "path")
    if path not in ("primal", "diff"):
        raise ValueError(f"path {path!r} is not 'primal' or 'diff'")
    return SiteKey(
        parent=parent,
        path=path,
        label=_field(body, "label"),
        ordinal=_field(body, "ordinal"),
    )


def _decision_to_json(decision: SiteDecision) -> dict[str, Any]:
    """Serialise one site decision."""
    diff = None if decision.diff is None else _strategy_to_json(decision.diff)
    return {
        "key": _key_to_json(decision.key),
        "kind": decision.kind,
        "primal": _strategy_to_json(decision.primal),
        "diff": diff,
        "primal_reason": decision.primal_reason,
        "diff_reason": decision.diff_reason,
        "warnings": list(decision.warnings),
    }


def _decision_from_json(payload: Any) -> SiteDecision:
    """Inverse of :func:`_decision_to_json`.

    Raises
    ------
    ValueError
        ``payload`` is not an object, a decision field is missing, a nested
        key or strategy is invalid, or ``diff`` is ``StaticWhile``.
    """
    body = _object(payload, "decision")
    diff_payload = _field(body, "diff")
    if diff_payload is None:
        diff = None
    else:
        decoded = _strategy_from_json(diff_payload)
        if isinstance(decoded, StaticWhile):
            raise ValueError("'diff' cannot be StaticWhile")
        diff = decoded
    warnings = _field(body, "warnings")
    if not isinstance(warnings, list):
        raise ValueError("'warnings' is not a list")
    return SiteDecision(
        key=_key_from_json(_field(body, "key")),
        kind=_field(body, "kind"),
        primal=_strategy_from_json(_field(body, "primal")),
        diff=diff,
        primal_reason=_field(body, "primal_reason"),
        diff_reason=_field(body, "diff_reason"),
        warnings=tuple(warnings),
    )
