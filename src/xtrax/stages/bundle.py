"""StageBundle: Typed bag of optional callable stage slots."""

import types
import typing
from collections.abc import Callable
from typing import Union, get_args, get_origin

import equinox as eqx


def _is_callable_type(typ) -> bool:
    """Check if a type is Callable, Callable[...], or a Protocol with only __call__.

    Accepts:
    - collections.abc.Callable (bare)
    - Callable[...] (generic form with signature)
    - A typing.Protocol subclass that declares only __call__ as its callable member
    """
    # Check if it's collections.abc.Callable
    if typ is Callable:
        return True
    # Check if it's Callable[...] (generic form)
    origin = get_origin(typ)
    if origin is Callable:
        return True

    # Check if it's a Protocol with only __call__
    if isinstance(typ, type) and getattr(typ, "_is_protocol", False):
        # Get the set of members defined directly on this Protocol class,
        # excluding internal attributes and the __call__ method itself
        own_members = set(typ.__dict__) - {
            "__parameters__",
            "_is_protocol",
            "_is_runtime_protocol",
            "__doc__",
            "__module__",
            "__annotations__",
            "__dict__",
            "__weakref__",
            "__subclasshook__",
            "__init__",
            "__abstractmethods__",
            "_abc_impl",
            "__classcell__",
            "__protocol_attrs__",
            "__non_callable_proto_members__",
            "__firstlineno__",
            "__static_attributes__",
        }
        # A callable-shaped Protocol has exactly __call__ and nothing else
        if "__call__" in typ.__dict__ and own_members <= {"__call__"}:
            return True

    return False


class StageBundle(eqx.Module):
    """Typed bag of optional callable stage slots. Subclass and declare fields.

    Topology determined by non-None fields at Python dispatch level.
    WARNING: Do NOT call active_stages/has_stage inside JAX traces — Python-side only.
    No __call__ method — pure container.

    All user-declared fields must be Optional[Callable].
    """

    def __init_subclass__(cls, **kwargs):
        """Validate that all annotated fields are Optional[Callable].

        Also rejects untyped class attributes to enforce strict typing.
        """
        super().__init_subclass__(**kwargs)

        # Resolve annotations via get_type_hints rather than reading
        # cls.__annotations__ directly: PEP 563's postponed-evaluation
        # future import (see e.g. src/xtrax/cli/emit.py:8, commit 3630277)
        # makes a consuming module's raw annotations plain strings;
        # get_type_hints() resolves them against the class's own __module__
        # globals. Fields must be declared with names resolvable at module
        # scope (the normal case for every real usage) -- a field typed
        # against a class defined only inside a function body will raise a
        # NameError here, same as any other library that resolves postponed
        # annotations this way (dataclasses, etc.). That NameError is a
        # deliberate, loud failure, not a gap: silently falling back to raw
        # (unresolved, always-rejected) annotations would defeat the entire
        # point of this fix.
        try:
            annotations = typing.get_type_hints(cls, include_extras=True)
        except NameError as exc:
            raise TypeError(
                f"{cls.__name__}: could not resolve one or more field "
                f"annotations ({exc}). StageBundle subclass fields must "
                f"reference names importable at module scope."
            ) from exc

        # Validate all annotated fields are Optional[Callable]
        for field_name, field_type in annotations.items():
            # Check if the annotation is Optional[Callable]
            origin = get_origin(field_type)

            # Handle both typing.Union and types.UnionType (from | operator)
            if origin is Union or origin is types.UnionType:
                # Optional[X] is Union[X, None]; generalized: Union[X, Y, ..., None]
                args = get_args(field_type)
                # Count None types and extract non-None members
                none_count = sum(1 for a in args if a is type(None))
                non_none = [a for a in args if a is not type(None)]

                # Must have exactly one None and at least one Callable
                if none_count != 1 or len(non_none) < 1:
                    raise TypeError(
                        f"Field '{field_name}' in {cls.__name__} must be "
                        f"Optional[Callable], got {field_type}"
                    )

                # All non-None types must be Callable
                if not all(_is_callable_type(t) for t in non_none):
                    raise TypeError(
                        f"Field '{field_name}' in {cls.__name__} must be "
                        f"Optional[Callable], got {field_type}"
                    )
            else:
                # Not Optional[...]
                raise TypeError(
                    f"Field '{field_name}' in {cls.__name__} must be "
                    f"Optional[Callable], got {field_type}"
                )

        # Reject any untyped class attributes (not in __annotations__)
        # This enforces strict typing for all fields
        for attr_name in dir(cls):
            # Skip magic methods, inherited attributes, non-field stuff
            if attr_name.startswith("_"):
                continue
            # Skip methods from parent classes that we expect to be there
            if attr_name in ("active_stages", "has_stage"):
                continue
            # Check if this is a class attribute not in annotations
            if hasattr(cls, attr_name) and attr_name not in annotations:
                attr_value = getattr(cls, attr_name)
                # Only reject actual values (not methods or inherited stuff)
                # Methods won't be plain values, so check if it's data
                if not callable(attr_value) or isinstance(attr_value, (classmethod, staticmethod)):
                    # This is a class variable without a type annotation
                    # But we need to be careful not to flag inherited stuff
                    # Check if it's defined directly on this class
                    if attr_name in cls.__dict__:
                        raise TypeError(
                            f"Field '{attr_name}' in {cls.__name__} must be "
                            f"annotated with a type; all fields must be "
                            f"Optional[Callable]"
                        )

    def active_stages(self) -> list[str]:
        """Return field names with non-None callable values. Python-side only."""
        return [name for name, val in vars(self).items() if val is not None and callable(val)]

    def has_stage(self, name: str) -> bool:
        """Return True if named field is a non-None callable. Python-side only."""
        val = getattr(self, name, None)
        return val is not None and callable(val)
