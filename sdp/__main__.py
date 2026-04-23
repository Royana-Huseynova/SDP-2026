"""Allow `python -m sdp ...` to dispatch to the CLI."""
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
