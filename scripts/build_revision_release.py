#!/usr/bin/env python3
"""Assemble an offline, externally unpublished revision-v1 release candidate."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_release import main


if __name__ == "__main__":
    raise SystemExit(main())
