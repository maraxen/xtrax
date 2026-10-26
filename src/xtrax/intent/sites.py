"""Site session for one intent trace.

Plan mode records each site. Exec mode checks that the same call sequence
still matches those records. A :class:`Session` does not read ambient
configuration: the active session is whichever :func:`activate` installed
in the :mod:`contextvars` context.
"""

import inspect
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import jax

from xtrax.intent.errors import IntentError, SiteMismatchError
from xtrax.intent.plan import SiteDecision, SiteKey

_Frame = tuple[SiteKey, Literal["primal", "diff"]]
_OrdinalKey = tuple[SiteKey | None, Literal["primal", "diff"], str]

_CURRENT: "ContextVar[Session | None]" = ContextVar("xtrax.intent.session", default=None)


@dataclass(frozen=True)
class SiteRecord:
    """One visited intent site.

    ``fn`` is the closure-converted site function. It is stored for a later
    estimator and is excluded from equality and hashing.

    Attributes
    ----------
    key : SiteKey
        Identity of this visit, including its ordinal.
    kind : {"loop", "map"}
        Site kind.
    avals : tuple of jax.ShapeDtypeStruct
        Abstract values of the site arguments.
    length : int or None
        Trip count for a loop, or ``None`` when the site has no length.
    fn : Callable or None
        Closure-converted site function. Ignored by ``==`` and ``hash``.
    const_avals : tuple of jax.ShapeDtypeStruct
        Abstract values of constants hoisted from the site function. Empty
        when nothing was hoisted. Passed to the estimator as closed-over
        inputs, not as mapped arguments.
    xs_tree : PyTreeDef or None
        Pytree structure of a map's ``xs``. ``None`` on records that store
        only a flat leaf list.
    """

    key: SiteKey
    kind: Literal["loop", "map"]
    avals: tuple[jax.ShapeDtypeStruct, ...]
    length: int | None
    fn: Callable[..., object] | None = field(compare=False, hash=False)
    const_avals: tuple[jax.ShapeDtypeStruct, ...] = ()
    xs_tree: jax.tree_util.PyTreeDef | None = None


class Session:
    """Plan or exec visit of intent sites.

    Outside any :meth:`scope`, an entered site has ``parent is None`` and
    ``path == "primal"``. :meth:`scope` nests: the site's parent is the key
    passed to the innermost scope, and its path is that scope's path.
    Ordinals count returned visits per ``(parent, path, label)`` from 0, in
    both modes, so an exec-mode key matches the plan-mode key for the same
    call sequence. A visit that raises keeps the previous count.

    Attributes
    ----------
    mode : {"plan", "exec"}
        ``"plan"`` appends a :class:`SiteRecord`. ``"exec"`` only checks one.
    records : list of SiteRecord
        Planned sites. A fresh list when constructed with ``records=None``.
    decisions : Mapping or None
        Decisions keyed by :class:`SiteKey`, or ``None`` when none were supplied.
    """

    mode: Literal["plan", "exec"]
    records: list[SiteRecord]
    decisions: Mapping[SiteKey, SiteDecision] | None

    def __init__(
        self,
        mode: Literal["plan", "exec"],
        records: list[SiteRecord] | None = None,
        decisions: Mapping[SiteKey, SiteDecision] | None = None,
    ) -> None:
        """Start a session.

        Parameters
        ----------
        mode : {"plan", "exec"}
            Whether :meth:`enter_site` records or checks.
        records : list of SiteRecord or None
            Planned records. ``None`` installs a new empty list, so sessions
            do not share a default list.
        decisions : Mapping or None
            Site decisions. ``None`` makes :meth:`decision_for` return ``None``.

        Returns
        -------
        None
        """
        self.mode = mode
        self.records = [] if records is None else records
        self.decisions = decisions
        self._frames: list[_Frame] = []
        self._ordinals: dict[_OrdinalKey, int] = {}
        self._label_files: dict[str, str] = {}

    def enter_site(
        self,
        kind: Literal["loop", "map"],
        label: str,
        avals: tuple[jax.ShapeDtypeStruct, ...],
        length: int | None,
        fn: Callable[..., object] | None,
        const_avals: tuple[jax.ShapeDtypeStruct, ...] = (),
        xs_tree: jax.tree_util.PyTreeDef | None = None,
    ) -> SiteKey:
        """Enter one site and return its key.

        The key's parent and path come from the scope stack. The ordinal is
        the next count for that parent, path, and label. Plan mode appends a
        :class:`SiteRecord`. Exec mode appends nothing: it looks the key up
        in ``records`` and checks that each abstract value has the same shape
        and dtype. The count advances only when this method returns the key.

        Parameters
        ----------
        kind : {"loop", "map"}
            Site kind stored on the plan record.
        label : str
            User name or ``module:line`` of the call.
        avals : tuple of jax.ShapeDtypeStruct
            Abstract values traced at this call.
        length : int or None
            Loop length, or ``None``.
        fn : Callable or None
            Closure-converted site function, or ``None``.
        const_avals : tuple of jax.ShapeDtypeStruct
            Hoisted constants. Stored on the plan record and ignored by the
            exec-mode shape check.
        xs_tree : PyTreeDef or None
            Structure of a map's ``xs``. Stored on the plan record so the
            estimator can rebuild the pytree.

        Returns
        -------
        SiteKey
            Key of this visit. In exec mode this is the planned key.

        Raises
        ------
        SiteMismatchError
            Exec mode, and the key is absent (``constraint`` is ``"site not
            in plan"``) or any shape, dtype, or aval count differs
            (``constraint`` is ``"planned <avals>, traced <avals>"``). The
            error names ``str(key)`` and ``key.path``.
        ValueError
            ``mode`` is not ``"plan"`` or ``"exec"``.
        """
        key = self._allocate_key(label)
        if self.mode == "plan":
            self.records.append(SiteRecord(key, kind, avals, length, fn, const_avals, xs_tree))
        elif self.mode == "exec":
            self._check_planned(key, avals)
        else:
            raise ValueError(f"mode {self.mode!r} is not 'plan' or 'exec'")
        counter: _OrdinalKey = (key.parent, key.path, key.label)
        self._ordinals[counter] = key.ordinal + 1
        return key

    def decision_for(self, key: SiteKey) -> SiteDecision | None:
        """Return the decision recorded for ``key``.

        Parameters
        ----------
        key : SiteKey
            Site to look up.

        Returns
        -------
        SiteDecision or None
            ``decisions[key]`` when ``decisions`` was supplied and contains
            ``key``. ``None`` when ``decisions`` is ``None`` or ``key`` is
            absent.
        """
        if self.decisions is None:
            return None
        return self.decisions.get(key)

    def push(self, key: SiteKey, path: Literal["primal", "diff"]) -> None:
        """Make ``key`` the parent of sites entered until :meth:`pop`.

        Parameters
        ----------
        key : SiteKey
            Enclosing site. Nested :meth:`enter_site` calls use it as
            ``parent``.
        path : {"primal", "diff"}
            Path those nested calls record on their keys.

        Returns
        -------
        None
        """
        self._frames.append((key, path))

    def pop(self) -> None:
        """Drop the innermost :meth:`push`.

        Parameters
        ----------
        self : Session
            Session whose scope stack is popped.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            The scope stack is empty.
        """
        if not self._frames:
            raise RuntimeError("scope stack is empty")
        self._frames.pop()

    @contextmanager
    def scope(self, key: SiteKey, path: Literal["primal", "diff"]) -> Iterator[None]:
        """Push ``(key, path)`` and pop it on the way out, including on error.

        Parameters
        ----------
        key : SiteKey
            Enclosing site, forwarded to :meth:`push`.
        path : {"primal", "diff"}
            Path of sites entered in this block.

        Returns
        -------
        None
            Yields nothing. The stack frame is removed when the block ends.
        """
        self.push(key, path)
        try:
            yield
        finally:
            self.pop()

    def _claim_label(self, label: str, filename: str) -> None:
        """Record that ``filename`` produced the call-site ``label``.

        Raises
        ------
        IntentError
            A different file already produced ``label`` in this session.
        """
        first = self._label_files.setdefault(label, filename)
        if first != filename:
            constraint = (
                f"label {label!r} comes from {first} and {filename}; "
                "pass name= to one of the call sites"
            )
            raise IntentError(
                f"call-site label {label!r} is shared by two files",
                site=label,
                path="primal",
                constraint=constraint,
            )

    def _allocate_key(self, label: str) -> SiteKey:
        """Reserve the next key for ``label`` without storing ``ordinal + 1``."""
        if self._frames:
            parent, path = self._frames[-1]
        else:
            parent, path = None, "primal"
        counter: _OrdinalKey = (parent, path, label)
        ordinal = self._ordinals.get(counter, 0)
        return SiteKey(parent=parent, path=path, label=label, ordinal=ordinal)

    def _check_planned(self, key: SiteKey, avals: tuple[jax.ShapeDtypeStruct, ...]) -> None:
        """Raise when ``key`` was not planned or its avals differ."""
        planned = _record_by_key(self.records).get(key)
        if planned is None:
            raise SiteMismatchError(
                "site not in plan",
                site=str(key),
                path=key.path,
                constraint="site not in plan",
            )
        if not _same_avals(planned.avals, avals):
            constraint = f"planned {planned.avals}, traced {avals}"
            raise SiteMismatchError(
                constraint,
                site=str(key),
                path=key.path,
                constraint=constraint,
            )


def current_session(*, default: Session | None = None) -> Session | None:
    """Return the session installed by :func:`activate`, if any.

    Parameters
    ----------
    default : Session or None
        Returned when no session is active. Defaults to ``None``. A session
        installed by :func:`activate` is returned instead of ``default``.

    Returns
    -------
    Session or None
        The active session, or ``default`` when none is active.
    """
    return _CURRENT.get(default)


@contextmanager
def activate(session: Session) -> Iterator[Session]:
    """Install ``session`` as :func:`current_session` for the block.

    The previous session is restored on the way out, including when the
    block raises. Nested :func:`activate` calls restore the outer session.

    Parameters
    ----------
    session : Session
        Session to install.

    Returns
    -------
    Session
        Yields ``session``.
    """
    token = _CURRENT.set(session)
    try:
        yield session
    finally:
        _CURRENT.reset(token)


def call_label(name: str | None, stacklevel: int = 1) -> str:
    """Return ``name``, or ``module:lineno`` of a caller frame.

    ``stacklevel=1`` is the direct caller of :func:`call_label`. Each
    greater level walks one frame further up. ``module`` is the caller's
    ``__name__``, so ``models/encoder/layers.py:42`` and
    ``models/decoder/layers.py:42`` read ``models.encoder.layers:42`` and
    ``models.decoder.layers:42`` and a label-keyed override reaches one of
    them. A frame with no usable ``__name__`` falls back to the file's base
    name.

    With a session active, a label built this way is also registered against
    the caller's file. A second file that produces the same label (for
    example two scripts that both run as ``__main__`` and call at one line)
    raises :class:`~xtrax.intent.errors.IntentError`, because an override
    keyed by that label would reach both sites.

    Parameters
    ----------
    name : str or None
        User name of the call. Returned unchanged when it is not ``None``,
        including when it is empty. A user name is not checked for origin:
        sharing one name across call sites is how one override reaches
        several of them.
    stacklevel : int
        Frames above :func:`call_label` to read when ``name`` is ``None``.
        ``1`` selects the direct caller.

    Returns
    -------
    str
        ``name`` when it is not ``None``. Otherwise ``f"{module}:{lineno}"``.

    Raises
    ------
    ValueError
        ``name`` is ``None`` and ``stacklevel`` is less than 1, or the walk
        passes the top of the stack.
    RuntimeError
        ``name`` is ``None`` and the interpreter has no caller frame.
    IntentError
        ``name`` is ``None``, a session is active, and the label was already
        produced by a different file in that session.
    """
    if name is not None:
        return name
    if stacklevel < 1:
        raise ValueError(f"stacklevel must be >= 1, got {stacklevel}")
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("call_label could not read the caller frame")
    for _ in range(stacklevel):
        parent = frame.f_back
        if parent is None:
            raise ValueError(f"stacklevel {stacklevel} is above the top of the stack")
        frame = parent
    filename = frame.f_code.co_filename
    module = frame.f_globals.get("__name__")
    if not isinstance(module, str) or not module:
        module = os.path.basename(filename)
    label = f"{module}:{frame.f_lineno}"
    session = current_session()
    if session is not None:
        session._claim_label(label, filename)
    return label


def _leaf_struct(leaf: Any) -> jax.ShapeDtypeStruct:
    """Abstract value of one leaf, including a Python scalar.

    Parameters
    ----------
    leaf : array, tracer, or Python scalar
        Value to summarise. Python scalars are accepted; their weak type is
        kept.

    Returns
    -------
    jax.ShapeDtypeStruct
        Shape, dtype, and ``weak_type`` from :func:`jax.typeof`.
    """
    aval = jax.typeof(leaf)
    return jax.ShapeDtypeStruct(
        tuple(aval.shape),
        aval.dtype,
        weak_type=bool(aval.weak_type),
    )


def _record_by_key(records: list[SiteRecord]) -> dict[SiteKey, SiteRecord]:
    """Index records by key. The first record wins when keys repeat."""
    indexed: dict[SiteKey, SiteRecord] = {}
    for record in records:
        indexed.setdefault(record.key, record)
    return indexed


def _same_avals(
    planned: tuple[jax.ShapeDtypeStruct, ...],
    traced: tuple[jax.ShapeDtypeStruct, ...],
) -> bool:
    """Compare aval sequences by length, shape, and dtype."""
    if len(planned) != len(traced):
        return False
    return all(
        left.shape == right.shape and left.dtype == right.dtype
        for left, right in zip(planned, traced, strict=True)
    )
