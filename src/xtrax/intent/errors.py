"""Error types for xtrax.intent resolution.

Unsatisfiable contexts and site mismatches name the site, path, and
constraint before any execution starts.
"""

from __future__ import annotations


class IntentError(Exception):
    """Base error for intent resolution failures.

    Attributes:
        site: User name or ``file:line`` of the failing site.
        path: Resolution path (for example ``primal`` or ``diff``).
        constraint: The constraint that could not be met.
    """

    def __init__(self, message: str, *, site: str, path: str, constraint: str) -> None:
        """Store site, path, and constraint and include all three in the message.

        Parameters
        ----------
        message : str
            What failed.
        site : str
            User name or ``file:line`` of the failing site.
        path : str
            Resolution path (for example ``primal`` or ``diff``).
        constraint : str
            The constraint that could not be met.

        Returns
        -------
        None
        """
        self.message = message
        self.site = site
        self.path = path
        self.constraint = constraint
        super().__init__(f"{message} (site={site}, path={path}, constraint={constraint})")

    def __reduce__(self) -> tuple[object, ...]:
        """Rebuild from the keyword fields so the error survives pickling.

        Parameters
        ----------
        self : IntentError
            Instance being pickled.

        Returns
        -------
        tuple
            ``(rebuild, args)`` for :mod:`pickle`.
        """
        return (
            _rebuild,
            (type(self), self.message, self.site, self.path, self.constraint),
        )


def _rebuild(
    cls: type[IntentError], message: str, site: str, path: str, constraint: str
) -> IntentError:
    """Reconstruct an intent error from its fields (pickle support).

    Parameters
    ----------
    cls : type[IntentError]
        Concrete error class.
    message : str
        What failed.
    site : str
        Failing site.
    path : str
        Resolution path.
    constraint : str
        Unmet constraint.

    Returns
    -------
    IntentError
        The rebuilt error.
    """
    return cls(message, site=site, path=path, constraint=constraint)


class SiteMismatchError(IntentError):
    """Raised when a site decision does not match the requested path."""


class UnsatisfiableContextError(IntentError):
    """Raised when no strategy satisfies the caller's context."""
