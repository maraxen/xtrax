"""CLI-layer error types for xtrax.

This module defines the exception hierarchy for errors that occur during
command-line interface operations, including argument parsing, function
loading, and shape specification.
"""

from __future__ import annotations


class CLIError(Exception):
    """Base exception for all CLI-layer errors.

    This is the root of the exception hierarchy for errors that occur during
    command-line interface operations. Specific CLI errors inherit from this
    class to enable unified exception handling and error reporting.
    """

    pass


class CLIImportError(CLIError):
    """Raised when a function import path cannot be resolved.

    This error is raised by the loader (entrypoint T1) when the CLI is
    unable to load a function from the specified import path. This typically
    occurs when a module cannot be found, a symbol does not exist in the
    module, or there are import-time errors in the target module.
    """

    pass


class ShapeParseError(CLIError):
    """Raised when a shape specification string is malformed.

    This error is raised by the shapes parser (entrypoint T2) when a shape
    string cannot be parsed into a valid shape specification. Shape strings
    follow a specific format, and this error indicates the input did not
    conform to that format.
    """

    pass


class ResumeError(CLIError):
    """Raised when checkpoint resumption fails due to missing or invalid state/manifest."""

    pass

