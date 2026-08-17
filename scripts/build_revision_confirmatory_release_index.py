#!/usr/bin/env python3
"""Build/check the legacy confirmatory-index compatibility artifact.

This is not the active code-and-data release assembler.
"""

from rankcloak.revision_release_index import main


if __name__ == "__main__":
    raise SystemExit(main())
