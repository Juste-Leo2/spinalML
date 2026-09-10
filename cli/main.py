# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import sys
from pathlib import Path

# Add the current directory to sys.path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.resolve()))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from spinalml_cli.cli import app

if __name__ == "__main__":
    app()
