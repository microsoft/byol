# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Entry point for ``python -m byol.train``.

Usage:
    python -m byol.train sft --model <path> --dataset <name>
    python -m byol.train merge --base-model <path> --adapter <path> --output <dir>
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
