import sys
from pathlib import Path

# Add the current directory to sys.path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from spinalml_cli.cli import app

if __name__ == "__main__":
    app()
