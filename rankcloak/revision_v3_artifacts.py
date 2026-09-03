"""Git-portable artifact inventory rules for the revision-V3 handoff.

The checked-in artifact manifest must describe exactly what a clean clone can
contain. When an output directory lives inside the repository, inventory the
union of tracked files and non-ignored untracked files. This excludes local
execution logs covered by ``.gitignore`` while still allowing a finalizer to
manifest newly generated results before they are staged. External output
directories have no Git portability contract, so all regular files are used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDED_ARTIFACTS = frozenset(
    {"artifact_manifest.csv", "provenance/validation_report.json"}
)


def _regular_files(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()})


def portable_artifact_files(
    output: Path,
    project_root: Path,
    *,
    excluded: Iterable[str] = DEFAULT_EXCLUDED_ARTIFACTS,
) -> list[Path]:
    """Return files that belong in a clean-clone-portable artifact manifest."""

    output = output.resolve()
    project_root = project_root.resolve()
    excluded_set = {str(item) for item in excluded}
    try:
        output_relative = output.relative_to(project_root)
    except ValueError:
        candidates = _regular_files(output.rglob("*"))
    else:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                output_relative.as_posix(),
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "could not enumerate Git-visible V3 artifacts: "
                + completed.stderr.decode("utf-8", "replace").strip()
            )
        candidates = []
        for encoded in completed.stdout.split(b"\0"):
            if not encoded:
                continue
            candidate = (project_root / encoded.decode("utf-8")).resolve()
            try:
                candidate.relative_to(output)
            except ValueError as exc:
                raise RuntimeError(
                    "Git artifact inventory escaped the output directory"
                ) from exc
            if candidate.is_file():
                candidates.append(candidate)
        candidates = _regular_files(candidates)

    return [
        path
        for path in candidates
        if path.relative_to(output).as_posix() not in excluded_set
    ]


__all__ = ["DEFAULT_EXCLUDED_ARTIFACTS", "portable_artifact_files"]
