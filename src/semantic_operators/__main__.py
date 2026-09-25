"""``python -m semantic_operators`` is the ``semop`` command."""

import sys

from .interfaces.cli import main

sys.exit(main())
