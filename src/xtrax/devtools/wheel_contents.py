"""Which repo paths ship in xtrax's wheel, read from pyproject.toml (#5036, #5090).

Gates that reason about what a consumer can import need the shipped/non-shipped
split, and it is already declared once -- `[tool.hatch.build.targets.wheel]`'s
`packages` and `exclude`. Reading it here keeps that declaration the single source,
rather than a second hand-maintained list inside each gate.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class WheelContents:
    packages: tuple[str, ...]  # repo-relative posix dirs, e.g. "src/xtrax"
    exclude: tuple[str, ...]  # repo-relative posix paths or dirs, e.g. "src/xtrax/devtools"

    def ships(self, rel_path: str) -> bool:
        """True iff the repo-relative file `rel_path` is inside the built wheel."""
        path = PurePosixPath(rel_path)

        def under(prefix: str) -> bool:
            p = PurePosixPath(prefix)
            return path == p or p in path.parents

        return any(under(pkg) for pkg in self.packages) and not any(
            under(ex) for ex in self.exclude
        )


def load_wheel_contents(pyproject: Path | dict) -> WheelContents:
    """Parse the hatch wheel target; `pyproject` is a path or an already-loaded dict."""
    data = pyproject if isinstance(pyproject, dict) else tomllib.loads(pyproject.read_text())
    wheel = (
        data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    )
    packages = tuple(str(p).rstrip("/") for p in wheel.get("packages", []))
    if not packages:
        msg = "pyproject.toml declares no [tool.hatch.build.targets.wheel] packages"
        raise ValueError(msg)
    exclude = tuple(str(p).rstrip("/") for p in wheel.get("exclude", []))
    globbed = [p for p in (*packages, *exclude) if any(c in p for c in "*?[")]
    if globbed:
        # Hatch reads these gitignore-style; a prefix match would silently misclassify.
        msg = f"wheel packages/exclude use glob patterns this reader does not model: {globbed}"
        raise ValueError(msg)
    return WheelContents(packages=packages, exclude=exclude)
