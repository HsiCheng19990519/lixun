"""
Stage 5 entrypoint.

This wraps the Stage 4 CLI but always enables Stage 5 helpers (auto file writes,
default hiking prompt/output paths, and search usage reporting).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from devmate import cli


def _inject_stage5_flags(argv: list[str]) -> list[str]:
    """
    Ensure the Stage 5 switches are present so behavior matches Stage 4 plus extras.
    """
    args = list(argv)
    if "--stage5" not in args:
        args.append("--stage5")
    return args


def _resolve_output_dir(argv: list[str]) -> Path:
    """
    Infer the output directory that Stage 5 will use, so we can back up any existing contents.
    """
    for i, arg in enumerate(argv):
        if arg in {"--output-dir", "--output_dir"}:
            if i + 1 < len(argv):
                return Path(argv[i + 1]).expanduser()
        elif arg.startswith("--output-dir="):
            return Path(arg.split("=", 1)[1]).expanduser()
        elif arg.startswith("--output_dir="):
            return Path(arg.split("=", 1)[1]).expanduser()
    return Path(cli.DEFAULT_STAGE5_OUTPUT_DIR).expanduser()


def _backup_existing_dir(path: Path) -> None:
    """
    If the target output directory already exists, rename it to a timestamped backup.
    """
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.parent / f"{path.name}.bak-{timestamp}"
    counter = 1
    while backup.exists():
        backup = path.parent / f"{path.name}.bak-{timestamp}-{counter}"
        counter += 1
    path.rename(backup)
    print(f"[stage5] Existing output backed up: {backup}")


def main() -> None:
    argv = _inject_stage5_flags(sys.argv[1:])
    output_dir = _resolve_output_dir(argv)
    _backup_existing_dir(output_dir)
    sys.argv = [sys.argv[0]] + argv
    cli.main()


if __name__ == "__main__":
    main()
