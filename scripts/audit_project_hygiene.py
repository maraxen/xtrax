#!/usr/bin/env python3
"""Distribution N8 project hygiene gate — README/CHANGELOG/CITATION (#1460).

Also covers #4969's group/extra alias contract (B3) and declared-vs-imported
dependency contract (B5): a shared `dev`/`eda` name declared in both
`[dependency-groups]` and `[project.optional-dependencies]` must be a single-element
alias (`["xtrax[<name>]"]`), and every runtime dependency/import must resolve in both
directions between `pyproject.toml` and `src/`.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "project_hygiene.toml"


def parse_init_version(init_path: Path, *, attribute: str = "__version__") -> str:
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == attribute:
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                raise ValueError(f"{init_path}:{node.lineno}: {attribute} must be a string literal")
    raise ValueError(f"{attribute} assignment not found in {init_path}")


@dataclass(frozen=True)
class ProjectHygieneConfig:
    version: str
    version_source: str
    version_attribute: str
    forbidden_root_paths: tuple[str, ...]
    required_files: tuple[str, ...]
    min_readme_bytes: int
    readme_markers: tuple[str, ...]
    changelog_markers: tuple[str, ...]
    citation_keys: tuple[str, ...]
    pyproject_urls: tuple[str, ...]
    import_name_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)


def load_project_hygiene_config(config_path: Path) -> ProjectHygieneConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    hygiene = data.get("hygiene")
    if not isinstance(hygiene, dict):
        raise ValueError(f"missing [hygiene] section in {config_path}")

    version = hygiene.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("hygiene.version must be a non-empty string")

    version_source = hygiene.get("version_source")
    if not isinstance(version_source, str) or not version_source:
        raise ValueError("hygiene.version_source must be a non-empty string")

    version_attribute = hygiene.get("version_attribute")
    if not isinstance(version_attribute, str) or not version_attribute:
        raise ValueError("hygiene.version_attribute must be a non-empty string")

    forbidden = hygiene.get("forbidden_root_paths", [])
    if not isinstance(forbidden, list):
        raise ValueError("hygiene.forbidden_root_paths must be a list")

    required = hygiene.get("required_files")
    if not isinstance(required, list) or not required:
        raise ValueError("hygiene.required_files must be a non-empty list")

    min_readme = hygiene.get("min_readme_bytes")
    if not isinstance(min_readme, int) or min_readme <= 0:
        raise ValueError("hygiene.min_readme_bytes must be a positive integer")

    def _markers(section: str) -> tuple[str, ...]:
        table = hygiene.get(section, {})
        if not isinstance(table, dict):
            raise ValueError(f"hygiene.{section} must be a table")
        values = table.get("markers") or table.get("keys") or table.get("required")
        if not isinstance(values, list) or not values:
            raise ValueError(f"hygiene.{section} list must be non-empty")
        return tuple(str(item) for item in values)

    readme_markers = _markers("readme_markers")
    changelog_markers = _markers("changelog_markers")

    citation_table = hygiene.get("citation_keys", {})
    if not isinstance(citation_table, dict):
        raise ValueError("hygiene.citation_keys must be a table")
    citation_keys = citation_table.get("keys")
    if not isinstance(citation_keys, list) or not citation_keys:
        raise ValueError("hygiene.citation_keys.keys must be a non-empty list")

    urls_table = hygiene.get("pyproject_urls", {})
    if not isinstance(urls_table, dict):
        raise ValueError("hygiene.pyproject_urls must be a table")
    pyproject_urls = urls_table.get("required")
    if not isinstance(pyproject_urls, list) or not pyproject_urls:
        raise ValueError("hygiene.pyproject_urls.required must be a non-empty list")

    overrides_table = data.get("import_name_overrides", {})
    if not isinstance(overrides_table, dict):
        raise ValueError("[import_name_overrides] must be a table")
    import_name_overrides: dict[str, tuple[str, ...]] = {}
    for import_name, dist_names in overrides_table.items():
        if not isinstance(dist_names, list) or not dist_names:
            raise ValueError(f"import_name_overrides.{import_name} must be a non-empty list")
        import_name_overrides[import_name] = tuple(str(item) for item in dist_names)

    return ProjectHygieneConfig(
        version=version,
        version_source=version_source,
        version_attribute=version_attribute,
        forbidden_root_paths=tuple(str(item) for item in forbidden),
        required_files=tuple(str(item) for item in required),
        min_readme_bytes=min_readme,
        readme_markers=readme_markers,
        changelog_markers=changelog_markers,
        citation_keys=tuple(str(item) for item in citation_keys),
        pyproject_urls=tuple(str(item) for item in pyproject_urls),
        import_name_overrides=import_name_overrides,
    )


def _parse_citation_version(citation_path: Path) -> str | None:
    text = citation_path.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*['\"]?([^'\"\n]+)", text, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_skill_version(skill_path: Path) -> str | None:
    """Return a SKILL.md's declared `xtrax_version:`, or None if it declares none.

    Only the frontmatter block is searched. A skill that declares no version is not a
    failure -- the marker is optional, and a skill without one cannot go stale.
    """
    text = skill_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    frontmatter = text if end == -1 else text[:end]
    match = re.search(r"^xtrax_version:\s*['\"]?([^'\"\n]+)", frontmatter, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_citation_keys(citation_path: Path) -> set[str]:
    keys: set[str] = set()
    for line in citation_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" "):
            keys.add(line.split(":", 1)[0].strip())
    return keys


def _normalize_dist_name(name: str) -> str:
    """PEP 503 normalization: case- and separator-insensitive distribution names."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement_name(requirement: str) -> str:
    """Extract the bare distribution name from a PEP 508 requirement string.

    Handles version specifiers, extras (`foo[bar]>=1`), and environment markers
    (`foo>=1; python_version<'4'`) -- only the leading name is needed here.
    """
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)", requirement)
    if match is None:
        raise ValueError(f"cannot parse requirement name from {requirement!r}")
    return match.group(1)


def check_group_extra_aliases(pyproject: dict) -> list[str]:
    """B3: a name declared in BOTH tables must be a single-element `xtrax[<name>]` alias.

    Disjointness is not the rule -- `dev`/`eda` are deliberately declared in both
    `[dependency-groups]` and `[project.optional-dependencies]` on purpose. What's
    forbidden is the two copies drifting apart (the #4969 root cause): a shared name's
    group value must be exactly `["xtrax[<name>]"]`. A name present in only one table
    is unconstrained (e.g. `docs`, which is group-only).
    """
    failures: list[str] = []
    groups = pyproject.get("dependency-groups", {})
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    if not isinstance(groups, dict) or not isinstance(extras, dict):
        return failures

    shared = sorted(set(groups) & set(extras))
    for name in shared:
        expected = [f"xtrax[{name}]"]
        actual = groups.get(name)
        if actual != expected:
            failures.append(
                f"pyproject.toml [dependency-groups].{name} must be exactly "
                f"{expected!r} (an alias to [project.optional-dependencies].{name}); "
                f"got {actual!r}"
            )
    return failures


_IMPORT_ERROR_LIKE_HANDLER_NAMES = {"ImportError", "ModuleNotFoundError", "Exception"}


def _handler_catches_import_error(handler: ast.excepthandler) -> bool:
    """True for a bare `except:`, or a handler naming ImportError/ModuleNotFoundError/
    Exception (directly or in a tuple, e.g. `except (ImportError, OSError):`)."""
    if handler.type is None:
        return True
    candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for exc in candidates:
        name = None
        if isinstance(exc, ast.Name):
            name = exc.id
        elif isinstance(exc, ast.Attribute):
            name = exc.attr
        if name in _IMPORT_ERROR_LIKE_HANDLER_NAMES:
            return True
    return False


def _handler_suppresses(handler: ast.excepthandler) -> bool:
    """True when the handler cannot let an exception escape, i.e. contains no `raise`.

    A handler that catches ImportError only to re-raise it as something friendlier --
    `except ImportError as e: raise RuntimeError("pip install foo") from e` -- still
    fails at the same point in a consumer's install, just with a different exception
    type. Exempting it would be exactly the "green over a false fact" this gate exists
    to catch, so a handler that can raise earns no exemption.
    """
    return not any(isinstance(node, ast.Raise) for stmt in handler.body for node in ast.walk(stmt))


class _GuardedImportVisitor(ast.NodeVisitor):
    """Marks Import/ImportFrom nodes lexically inside a `try` block's `body` whose
    handlers catch ImportError/ModuleNotFoundError/bare-except AND cannot themselves
    raise -- such an import cannot fail a consumer's install, so B5 direction 2
    exempts it. A handler that re-raises earns no exemption: it fails at the same
    point, merely with a different exception type.

    Real call sites this protects: `src/xtrax/telemetry/record.py` (`cisternal`,
    `try: ... except ImportError: return None`) and `src/xtrax/telemetry/store.py`
    (`zstandard`, same pattern). Both genuinely suppress.

    `ast.walk()` loses parent links, so containment is tracked via an explicit guard
    stack during a manual recursive descent: only `ast.Try` gets custom
    body/handlers/orelse/finalbody handling (everything else falls through to
    `generic_visit`'s default traversal, so nesting inside `if`/`for`/`def`/etc.
    composes automatically). Only a try's own `body` is protected by its handlers --
    `orelse`/`finalbody` are visited with that try's own guard already popped. Nested
    tries compose naturally via the stack: an inner try without a matching handler is
    still exempted if an OUTER try's handler (still on the stack) matches, since the
    import is lexically within the outer try's body either way.
    """

    def __init__(self) -> None:
        self.guard_stack: list[bool] = []
        self.guarded_ids: set[int] = set()

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(
            _handler_catches_import_error(h) and _handler_suppresses(h) for h in node.handlers
        )
        self.guard_stack.append(guarded)
        for stmt in node.body:
            self.visit(stmt)
        self.guard_stack.pop()
        for handler in node.handlers:
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def _mark_if_guarded(self, node: ast.AST) -> None:
        if any(self.guard_stack):
            self.guarded_ids.add(id(node))

    def visit_Import(self, node: ast.Import) -> None:
        self._mark_if_guarded(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._mark_if_guarded(node)
        self.generic_visit(node)


def _collect_src_imports(src_root: Path) -> dict[str, list[tuple[Path, int, bool]]]:
    """Map every third-party top-level import name to its (file, lineno, is_guarded)
    occurrences.

    Walks every `ast.Import`/`ast.ImportFrom` node reachable via `ast.walk` -- i.e. at
    ANY depth, not just `tree.body`. This is load-bearing: `jaxlib`'s only appearance
    in the tree is `import jaxlib` inside a function body
    (`src/xtrax/profiling/record.py`), so a walk restricted to top-level statements
    would report a false positive on a dependency that is genuinely used.

    `is_guarded` reflects `_GuardedImportVisitor` -- computed once per file, since
    node identity (`id()`) is only meaningful while that file's tree is alive.
    """
    occurrences: dict[str, list[tuple[Path, int, bool]]] = {}
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guard_visitor = _GuardedImportVisitor()
        guard_visitor.visit(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                guarded = id(node) in guard_visitor.guarded_ids
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    occurrences.setdefault(top, []).append((path, node.lineno, guarded))
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    guarded = id(node) in guard_visitor.guarded_ids
                    top = node.module.split(".")[0]
                    occurrences.setdefault(top, []).append((path, node.lineno, guarded))
    return occurrences


def _dist_to_import_names(env_map: dict[str, list[str]]) -> dict[str, set[str]]:
    """Invert `importlib.metadata.packages_distributions()` to dist-name -> import names.

    `env_map` (import_name -> [dist_name, ...]) reflects the CURRENTLY INSTALLED
    environment, not a hand-maintained table -- avoiding a second copy of
    pyproject.toml's names that could itself drift out of sync (the #4969 failure
    class this whole gate exists to catch).
    """
    result: dict[str, set[str]] = {}
    for import_name, dists in env_map.items():
        for dist in dists:
            result.setdefault(_normalize_dist_name(dist), set()).add(import_name)
    return result


def _declared_names(pyproject: dict) -> set[str]:
    """Every normalized distribution name declared as a runtime dependency or extra."""
    project = pyproject.get("project", {})
    names: set[str] = set()
    for requirement in project.get("dependencies", []) or []:
        names.add(_normalize_dist_name(_requirement_name(requirement)))
    extras = project.get("optional-dependencies", {})
    if isinstance(extras, dict):
        for requirements in extras.values():
            if not isinstance(requirements, list):
                continue
            for requirement in requirements:
                names.add(_normalize_dist_name(_requirement_name(requirement)))
    return names


def check_dependencies_are_imported(
    root: Path,
    pyproject: dict,
    env_map: dict[str, list[str]],
) -> list[str]:
    """B5 direction 1: every `[project].dependencies` name is imported somewhere under src/."""
    failures: list[str] = []
    project = pyproject.get("project", {})
    dependencies = project.get("dependencies", []) or []
    if not dependencies:
        return failures

    src_imports = set(_collect_src_imports(root / "src").keys())
    dist_to_imports = _dist_to_import_names(env_map)

    for requirement in dependencies:
        req_name = _requirement_name(requirement)
        normalized = _normalize_dist_name(req_name)
        if normalized == "xtrax":
            continue
        candidates = dist_to_imports.get(normalized)
        if not candidates:
            # Not resolvable in the installed environment -- fall back to a
            # name-equality guess (hyphens/dots become underscores in import names)
            # rather than silently skipping verification.
            candidates = {normalized.replace("-", "_")}
        if not (candidates & src_imports):
            failures.append(
                f"dependency {req_name!r} (candidate import name(s) "
                f"{sorted(candidates)}) is declared in [project].dependencies but "
                "never imported anywhere under src/"
            )
    return failures


def check_imports_are_declared(
    root: Path,
    pyproject: dict,
    env_map: dict[str, list[str]],
    import_name_overrides: dict[str, tuple[str, ...]],
) -> list[str]:
    """B5 direction 2: every third-party import under src/ resolves to a declared name.

    Resolution order per import name (failing only when ALL three miss):
      1. the installed-environment map (`packages_distributions()`);
      2. normalised name-equality against every declared dependency/extra name;
      3. an explicit `[import_name_overrides]` entry in project_hygiene.toml, for the
         residue where the import name differs from the distribution name AND the
         distribution is not installed (e.g. `iree`, gated behind an unsynced extra).

    A lazy/optional import guarded behind an extra (e.g. `zarr`) is EXPECTED to
    resolve via a declared extra rather than `dependencies` -- that's success, not
    a failure.

    Applied LAST, only when all three steps above miss: an import lexically inside a
    `try` block whose handlers catch ImportError/ModuleNotFoundError/bare-except is
    exempted (see `_GuardedImportVisitor`). Such an import cannot raise
    ModuleNotFoundError at a consumer's install -- direction 2's whole rationale for
    existing -- so it is not the failure mode this check exists to catch. An
    UNGUARDED import that resolves nowhere still fails exactly as before; this
    exemption never weakens that.
    """
    failures: list[str] = []
    src_imports = _collect_src_imports(root / "src")
    stdlib = set(sys.stdlib_module_names)
    declared = _declared_names(pyproject)
    # import_name -> normalized dist names actually installed for it
    installed_dists_for_import: dict[str, set[str]] = {
        import_name: {_normalize_dist_name(d) for d in dists}
        for import_name, dists in env_map.items()
    }

    for import_name in sorted(src_imports):
        if import_name == "xtrax" or import_name in stdlib:
            continue

        env_dists = installed_dists_for_import.get(import_name, set())
        if env_dists & declared:
            continue
        if _normalize_dist_name(import_name) in declared:
            continue
        override_dists = import_name_overrides.get(import_name)
        if override_dists and ({_normalize_dist_name(d) for d in override_dists} & declared):
            continue

        # Guarded-import exemption -- applied last, only after all three resolution
        # steps missed. If EVERY occurrence of this import name is guarded, it's
        # fully exempt. If at least one occurrence is unguarded, report that one --
        # it's the one that can actually break a consumer's install.
        occurrences = src_imports[import_name]
        unguarded = [(p, ln) for (p, ln, guarded) in occurrences if not guarded]
        if not unguarded:
            continue

        first_path, first_line = unguarded[0]
        rel = first_path.relative_to(root)
        location = f"{rel}:{first_line}"

        if env_dists:
            failures.append(
                f"import {import_name!r} ({location}) resolves to installed "
                f"distribution(s) {sorted(env_dists)}, none of which are declared "
                "as a dependency or extra in pyproject.toml"
            )
        elif override_dists:
            failures.append(
                f"import {import_name!r} ({location}) has an "
                f"[import_name_overrides] entry {sorted(override_dists)}, but none "
                "of those distribution names are declared as a dependency or extra"
            )
        else:
            failures.append(
                f"import {import_name!r} ({location}) does not resolve to any "
                "declared dependency or extra: not in the installed environment, "
                "not name-equal to a declared name, and no [import_name_overrides] "
                "entry covers it"
            )

    # Self-cleaning check: an override entry resolvable via steps 1 or 2 is stale --
    # it should have been removed once its distribution became installed/resolvable,
    # keeping the hand-written table bounded to genuine un-installable residue.
    for import_name, override_dists in sorted(import_name_overrides.items()):
        env_dists = installed_dists_for_import.get(import_name, set())
        if env_dists & declared:
            failures.append(
                f"[import_name_overrides].{import_name} is stale: now resolvable "
                f"via the installed environment ({sorted(env_dists)}); remove the "
                "override"
            )
            continue
        if _normalize_dist_name(import_name) in declared:
            failures.append(
                f"[import_name_overrides].{import_name} is stale: now resolvable "
                "via name-equality against a declared name; remove the override"
            )

    return failures


def audit_project_hygiene(
    root: Path,
    config_path: Path,
) -> tuple[bool, list[str]]:
    config = load_project_hygiene_config(config_path)
    failures: list[str] = []

    for rel in config.required_files:
        path = root / rel
        if not path.is_file():
            failures.append(f"missing required file: {rel}")

    for rel in config.forbidden_root_paths:
        if (root / rel).exists():
            failures.append(f"forbidden root path present: {rel}")

    readme_path = root / "README.md"
    if readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8")
        if readme_path.stat().st_size < config.min_readme_bytes:
            failures.append(f"README.md too small ({readme_path.stat().st_size} bytes)")
        for marker in config.readme_markers:
            if marker not in readme_text:
                failures.append(f"README.md missing marker: {marker!r}")

    changelog_path = root / "CHANGELOG.md"
    if changelog_path.is_file():
        changelog_text = changelog_path.read_text(encoding="utf-8")
        for marker in config.changelog_markers:
            if marker not in changelog_text:
                failures.append(f"CHANGELOG.md missing marker: {marker!r}")

    citation_path = root / "CITATION.cff"
    init_path = root / config.version_source
    if citation_path.is_file() and init_path.is_file():
        present_keys = _parse_citation_keys(citation_path)
        missing_keys = [key for key in config.citation_keys if key not in present_keys]
        if missing_keys:
            failures.append("CITATION.cff missing keys: " + ", ".join(missing_keys))
        package_version = parse_init_version(
            init_path,
            attribute=config.version_attribute,
        )
        citation_version = _parse_citation_version(citation_path)
        if citation_version is None:
            failures.append("CITATION.cff missing version field")
        elif citation_version != package_version:
            failures.append(
                "CITATION.cff version "
                f"{citation_version!r} != {config.version_attribute} "
                f"{package_version!r}"
            )

    # agent_assets/skills/*/SKILL.md is a third version site, after __init__.py and
    # CITATION.cff. It was missed at 0.4.0a8 exactly as CITATION.cff would have been
    # without the check above: all three markers still read 0.4.0a7 after the release,
    # and nothing anywhere would have said so.
    if init_path.is_file():
        package_version = parse_init_version(init_path, attribute=config.version_attribute)
        for skill_path in sorted((root / "agent_assets" / "skills").glob("*/SKILL.md")):
            skill_version = _parse_skill_version(skill_path)
            if skill_version is not None and skill_version != package_version:
                rel = skill_path.relative_to(root)
                failures.append(
                    f"{rel} xtrax_version {skill_version!r} != "
                    f"{config.version_attribute} {package_version!r}"
                )

    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        readme = project.get("readme")
        if readme != "README.md":
            failures.append("pyproject.toml project.readme must be README.md")
        urls = project.get("urls", {})
        if not isinstance(urls, dict):
            failures.append("pyproject.toml missing [project.urls]")
        else:
            for key in config.pyproject_urls:
                if key not in urls:
                    failures.append(f"pyproject.toml missing project.urls.{key}")

        # B3 (#4969): dependency-groups/optional-dependencies alias contract.
        failures.extend(check_group_extra_aliases(data))

        # B5 (#4969): declared-vs-imported dependency contract, both directions.
        src_root = root / "src"
        if src_root.is_dir():
            env_map = importlib.metadata.packages_distributions()
            failures.extend(check_dependencies_are_imported(root, data, env_map))
            failures.extend(
                check_imports_are_declared(root, data, env_map, config.import_name_overrides)
            )

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to project_hygiene.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_project_hygiene(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
    )
    if passed:
        print("PASS: project hygiene gate")
        return 0

    print("FAIL: project hygiene gate", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
