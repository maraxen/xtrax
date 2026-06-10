"""StageBundle: Typed bag of optional callable stage slots."""

import types
from collections.abc import Callable
from typing import Union, get_args, get_origin

import equinox as eqx


def _is_callable_type(typ) -> bool:
    """Check if a type is Callable or Callable[...]."""
    # Check if it's collections.abc.Callable
    if typ is Callable:
        return True
    # Check if it's Callable[...] (generic form)
    origin = get_origin(typ)
    if origin is Callable:
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

        # Get type annotations from the class
        annotations = getattr(cls, "__annotations__", {})

        # Validate all annotated fields are Optional[Callable]
        for field_name, field_type in annotations.items():
            # Check if the annotation is Optional[Callable]
            origin = get_origin(field_type)

            # Handle both typing.Union and types.UnionType (from | operator)
            if origin is Union or origin is types.UnionType:
                # Optional[X] is Union[X, None]
                args = get_args(field_type)
                # Should have exactly 2 args: one Callable and one NoneType
                if len(args) != 2 or type(None) not in args:
                    raise TypeError(
                        f"Field '{field_name}' in {cls.__name__} must be "
                        f"Optional[Callable], got {field_type}"
                    )
                # Get the non-None type
                callable_type = args[0] if args[1] is type(None) else args[1]
                # Check if it's a Callable type
                if not _is_callable_type(callable_type):
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
                if not callable(attr_value) or isinstance(
                    attr_value, (classmethod, staticmethod)
                ):
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
        return [
            name
            for name, val in vars(self).items()
            if val is not None and callable(val)
        ]

    def has_stage(self, name: str) -> bool:
        """Return True if named field is a non-None callable. Python-side only."""
        val = getattr(self, name, None)
        return val is not None and callable(val)
